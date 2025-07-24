import re
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_together import TogetherLLM
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from db_connection import DatabaseConnection
from prompts import PromptManager


class NL2SQLState(TypedDict):
    """State for the NL2SQL workflow"""
    question: str
    db_schema: str
    relevant_tables: List[str]
    sql_query: str
    query_results: List[Dict]
    formatted_results: str
    explanation: str
    error_message: str
    context: str


class SQLChain:
    """Main SQL processing chain using LangGraph and Together AI"""
    
    def __init__(self, 
                 together_api_key: str, 
                 model_name: str = "meta-llama/Llama-3-70b-chat-hf",
                 temperature: float = 0.1):
        """
        Initialize the SQL processing chain
        """
        self.together_api_key = together_api_key
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self.llm = None
        self.db_connection = None
        self.prompt_manager = PromptManager()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.graph = None
        
        # Setup
        self._setup_llm()
        self._setup_graph()
    
    def _setup_llm(self):
        """Initialize the Together AI LLM"""
        self.llm = TogetherLLM(
            model=self.model_name,
            together_api_key=self.together_api_key,
            temperature=self.temperature,
            max_tokens=2000
        )
    
    def _setup_graph(self):
        """Setup the LangGraph workflow for NL2SQL processing"""
        workflow = StateGraph(NL2SQLState)
        
        # Add workflow nodes
        workflow.add_node("analyze_schema", self._analyze_schema)
        workflow.add_node("select_tables", self._select_relevant_tables)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql", self._validate_sql)
        workflow.add_node("execute_query", self._execute_query)
        workflow.add_node("format_results", self._format_results)
        workflow.add_node("explain_query", self._explain_query)
        
        # Define workflow edges
        workflow.add_edge("analyze_schema", "select_tables")
        workflow.add_edge("select_tables", "generate_sql")
        workflow.add_edge("generate_sql", "validate_sql")
        workflow.add_edge("validate_sql", "execute_query")
        workflow.add_edge("execute_query", "format_results")
        workflow.add_edge("format_results", "explain_query")
        
        # Set entry and finish points
        workflow.set_entry_point("analyze_schema")
        workflow.set_finish_point("explain_query")
        
        self.graph = workflow.compile()
    
    def set_database_connection(self, db_connection: DatabaseConnection):
        """Set the database connection for the SQL chain"""
        self.db_connection = db_connection
    
    def _get_conversation_context(self) -> str:
        """Get formatted conversation context from memory"""
        if not self.memory.chat_memory.messages:
            return "No previous conversation."
        
        context_parts = []
        for message in self.memory.chat_memory.messages[-5:]:  # Last 5 messages
            if hasattr(message, 'content'):
                role = "User" if message.type == "human" else "Assistant"
                context_parts.append(f"{role}: {message.content}")
        
        return "\n".join(context_parts)
    
    def _analyze_schema(self, state: NL2SQLState) -> NL2SQLState:
        """Analyze database schema and extract relevant information"""
        if not self.db_connection or not self.db_connection.db:
            state["error_message"] = "No database connection established"
            return state
        
        try:
            schema_info = self.db_connection.get_schema_info()
            state["db_schema"] = schema_info
            state["context"] = self._get_conversation_context()
            return state
        except Exception as e:
            state["error_message"] = f"Error analyzing schema: {str(e)}"
            return state
    
    def _select_relevant_tables(self, state: NL2SQLState) -> NL2SQLState:
        """Select relevant tables based on the question"""
        try:
            prompt = self.prompt_manager.get_prompt('table_selection')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run(
                question=state["question"], 
                schema=state["db_schema"]
            )
            
            # Parse table names from response
            if "NO_TABLES_FOUND" in response.upper():
                state["relevant_tables"] = []
            else:
                table_names = [name.strip() for name in response.split(",") if name.strip()]
                state["relevant_tables"] = table_names
            
            return state
        except Exception as e:
            state["error_message"] = f"Error selecting tables: {str(e)}"
            return state
    
    def _generate_sql(self, state: NL2SQLState) -> NL2SQLState:
        """Generate SQL query based on the question and relevant tables"""
        try:
            prompt = self.prompt_manager.get_prompt('sql_generation')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            response = chain.run(
                question=state["question"],
                schema=state["db_schema"],
                tables=", ".join(state["relevant_tables"]),
                context=state["context"]
            )
            
            # Clean and validate the SQL query
            sql_query = self._clean_sql_query(response)
            state["sql_query"] = sql_query
            return state
        except Exception as e:
            state["error_message"] = f"Error generating SQL: {str(e)}"
            return state
    
    def _validate_sql(self, state: NL2SQLState) -> NL2SQLState:
        """Validate the generated SQL query"""
        if not state["sql_query"]:
            state["error_message"] = "No SQL query to validate"
            return state
        
        try:
            # Basic validation using LLM
            prompt = self.prompt_manager.get_prompt('query_validation')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            validation_result = chain.run(
                schema=state["db_schema"],
                sql_query=state["sql_query"],
                question=state["question"]
            )
            
            if "INVALID" in validation_result.upper():
                state["error_message"] = f"Query validation failed: {validation_result}"
            
            return state
        except Exception as e:
            state["error_message"] = f"Error validating SQL: {str(e)}"
            return state
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and validate the generated SQL query"""
        # Remove any markdown formatting
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = re.sub(r'^sql\s*', '', sql_query, flags=re.IGNORECASE)
        
        # Remove extra whitespace and newlines
        sql_query = sql_query.strip()
        
        # Ensure it's a SELECT query only
        if not sql_query.upper().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
        
        # Check for prohibited keywords
        prohibited_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
        ]
        
        sql_upper = sql_query.upper()
        for keyword in prohibited_keywords:
            if keyword in sql_upper:
                raise ValueError(f"Prohibited SQL operation detected: {keyword}")
        
        return sql_query
    
    def _execute_query(self, state: NL2SQLState) -> NL2SQLState:
        """Execute the generated SQL query"""
        if not self.db_connection or not self.db_connection.engine:
            state["error_message"] = "No database connection available"
            return state
        
        if not state["sql_query"]:
            state["error_message"] = "No SQL query to execute"
            return state
        
        try:
            with self.db_connection.engine.connect() as connection:
                result = connection.execute(text(state["sql_query"]))
                
                # Convert results to list of dictionaries
                columns = result.keys()
                rows = result.fetchall()
                
                query_results = []
                for row in rows:
                    query_results.append(dict(zip(columns, row)))
                
                state["query_results"] = query_results
                return state
                
        except SQLAlchemyError as e:
            # Analyze error using LLM
            try:
                prompt = self.prompt_manager.get_prompt('error_analysis')
                chain = LLMChain(llm=self.llm, prompt=prompt)
                
                error_analysis = chain.run(
                    question=state["question"],
                    sql_query=state["sql_query"],
                    error_message=str(e),
                    schema=state["db_schema"]
                )
                
                state["error_message"] = f"SQL execution error: {str(e)}\n\nAnalysis: {error_analysis}"
            except:
                state["error_message"] = f"SQL execution error: {str(e)}"
            
            return state
        except Exception as e:
            state["error_message"] = f"Unexpected error during query execution: {str(e)}"
            return state
    
    def _format_results(self, state: NL2SQLState) -> NL2SQLState:
        """Format query results for display"""
        if state.get("error_message"):
            return state  # Skip formatting if there's an error
        
        if not state["query_results"]:
            state["formatted_results"] = "No results found for your query."
            return state
        
        try:
            # Use LLM to format results intelligently
            prompt = self.prompt_manager.get_prompt('result_formatting')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Convert results to string for prompt
            df = pd.DataFrame(state["query_results"])
            raw_results_str = df.to_string(index=False, max_rows=20)
            
            formatted_response = chain.run(
                question=state["question"],
                sql_query=state["sql_query"],
                raw_results=raw_results_str
            )
            
            # Also include basic table format as fallback
            result_count = len(state["query_results"])
            basic_format = f"Query returned {result_count} row(s):\n\n{raw_results_str}"
            
            if result_count > 20:
                basic_format += f"\n\n... (showing first 20 of {result_count} results)"
            
            state["formatted_results"] = f"{formatted_response}\n\n--- Raw Results ---\n{basic_format}"
            return state
            
        except Exception as e:
            # Fallback to basic formatting
            try:
                df = pd.DataFrame(state["query_results"])
                formatted_table = df.to_string(index=False, max_rows=50)
                result_count = len(state["query_results"])
                
                state["formatted_results"] = f"Query returned {result_count} row(s):\n\n{formatted_table}"
                if result_count > 50:
                    state["formatted_results"] += f"\n\n... (showing first 50 of {result_count} results)"
                
            except Exception as format_error:
                state["error_message"] = f"Error formatting results: {str(format_error)}"
            
            return state
    
    def _explain_query(self, state: NL2SQLState) -> NL2SQLState:
        """Generate explanation for the SQL query"""
        if state.get("error_message"):
            return state  # Skip explanation if there's an error
        
        try:
            prompt = self.prompt_manager.get_prompt('query_explanation')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            explanation = chain.run(
                question=state["question"],
                sql_query=state["sql_query"],
                schema=state["db_schema"],
                results_count=len(state["query_results"]) if state["query_results"] else 0
            )
            
            state["explanation"] = explanation.strip()
            return state
        except Exception as e:
            state["error_message"] = f"Error generating explanation: {str(e)}"
            return state
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return SQL query, results, and explanation
        
        Args:
            question: Natural language question about the database
            
        Returns:
            Dictionary containing results or error information