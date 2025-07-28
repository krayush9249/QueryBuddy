import re
import json
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import text
from db_connect import DatabaseConnection
from prompts import PromptManager
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state_schema import NL2SQLState


def analyze_schema(state: NL2SQLState, 
                  db_connection: DatabaseConnection) -> NL2SQLState:
    """Analyze database schema and extract relevant information"""
    if not db_connection or not db_connection.db:
        state["error_message"] = "No database connection established"
        return state
    
    try:
        schema_info = db_connection.get_schema_info()
        state["db_schema"] = schema_info
        return state
    except Exception as e:
        state["error_message"] = f"Error analyzing schema: {str(e)}"
        return state


class TableSelectionOutput(BaseModel):
    relevant_tables: List[str] = Field(
        ..., description="List of relevant table names."
    )

def select_relevant_tables(state: NL2SQLState,
                           llm,
                           prompt_manager: PromptManager) -> NL2SQLState:
    """Select relevant tables based on the question using structured output"""
    try:
        prompt = prompt_manager.get_prompt('table_selection')
        structured_llm = llm.with_structured_output(TableSelectionOutput)
        chain = prompt | structured_llm

        response = chain.invoke({
            "question": state["question"],
            "schema": state["db_schema"]
        })
        state["relevant_tables"] = response.relevant_tables
        return state

    except Exception as e:
        state["error_message"] = f"Error selecting tables: {str(e)}"
        return state


class SQLQueryValidator(BaseModel):
    sql_query: str

    @field_validator('sql_query', mode='before')
    @classmethod
    def clean_and_extract_sql(cls, v):
        """Extract SQL query from various formats"""
        content = str(v)
        
        # Step 1: Handle JSON format (most common case)
        try:
            if content.strip().startswith('{') and content.strip().endswith('}'):
                parsed = json.loads(content.strip())
                if 'sql_query' in parsed:
                    return parsed['sql_query']
        except json.JSONDecodeError:
            pass
        
        # Step 2: Regex JSON extraction
        match = re.search(r'"sql_query":\s*"([^"]*)"', content)
        if match:
            return match.group(1)
        
        # Step 3: Extract JSON from mixed text
        json_match = re.search(r'\{[^}]*"sql_query"[^}]*\}', content)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                extracted_sql = parsed.get('sql_query', '')
                if extracted_sql:
                    return extracted_sql
            except json.JSONDecodeError:
                pass
        
        # Step 4: Find SELECT statement with semicolon
        select_pattern = re.search(r'(SELECT\s+.*?;)', content, re.DOTALL | re.IGNORECASE)
        if select_pattern:
            return select_pattern.group(1)
        
        # If no valid SQL found, fail explicitly
        raise ValueError(f"Could not extract SQL query from input: {content[:100]}...")

    @field_validator('sql_query')
    @classmethod
    def must_be_select_query(cls, v):
        if not v or not v.upper().startswith("SELECT"):
            raise ValueError(f"Only SELECT queries are allowed. Got: {v}")
        return v

    @model_validator(mode='after')
    def check_prohibited_keywords(cls, values):
        sql = values.sql_query.upper()
        prohibited_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
        ]
        for keyword in prohibited_keywords:
            if keyword in sql:
                raise ValueError(f"Prohibited SQL operation detected: {keyword}")
        return values

def generate_sql(state: NL2SQLState, 
                llm,
                prompt_manager: PromptManager) -> NL2SQLState:
    """Generate SQL query based on the question and relevant tables"""
    try:
        prompt = prompt_manager.get_prompt('sql_generation')
        chain = prompt | llm
        
        # Get raw response from LLM
        raw_response = chain.invoke({
            "question": state["question"],
            "schema": state["db_schema"],
            "tables": ", ".join(state["relevant_tables"]),
            "sql_dialect": state['sql_dialect']
        })
        
        # Use validator to parse and validate the response
        validated_query = SQLQueryValidator(sql_query=raw_response.content)
        state["sql_query"] = validated_query.sql_query
        return state
   
    except Exception as e:
        state["error_message"] = f"Error generating SQL: {str(e)}"
        return state

def explain_query(state: NL2SQLState, 
                 llm,
                 prompt_manager: PromptManager) -> NL2SQLState:
    """Generate explanation for the SQL query"""
    if state.get("error_message"):
        return state  
    
    try:
        prompt = prompt_manager.get_prompt('query_explanation')
        chain = prompt | llm
        
        explanation = chain.invoke({
            "question": state["question"],
            "sql_query": state["sql_query"],
            "schema": state["db_schema"]
        })
        state["explanation"] = explanation.content.strip()
        return state
    
    except Exception as e:
        state["error_message"] = f"Error generating explanation: {str(e)}"
        return state
    
    
def execute_query(state: NL2SQLState, 
                  db_connection: DatabaseConnection) -> NL2SQLState:
    """Execute the generated SQL query using an agent"""

    if not db_connection or not db_connection.engine:
        state["error_message"] = "No database connection available"
        return state

    if not state["sql_query"]:
        state["error_message"] = "No SQL query to execute"
        return state

    try:
        with db_connection.engine.connect() as connection:
            result = connection.execute(text(state["sql_query"]))
            
            columns = result.keys()
            rows = result.fetchall()
            
            query_results = []
            for row in rows:
                query_results.append(dict(zip(columns, row)))
            
            state["query_results"] = query_results
        return state

    except Exception as e:
        state["error_message"] = f"Unexpected error during query execution: {str(e)}"
        return state
    

def format_results(state: NL2SQLState, 
                  llm,
                  prompt_manager: PromptManager) -> NL2SQLState:
    """Format query results for display"""
    if state.get("error_message"):
        return state 
    
    if not state["query_results"]:
        state["formatted_response"] = "No results found for your query."
        return state
    
    try:
        prompt = prompt_manager.get_prompt('result_formatting')
        chain = prompt | llm
        
        # Convert results to string for prompt
        df = pd.DataFrame(state["query_results"])
        raw_results_str = df.to_string(index=False, max_rows=25)
        
        # Include chat history for context if available
        chat_history_str = ""
        if state.get("chat_history"):
            recent_history = state["chat_history"][-4:]  # Last 2 exchanges
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        formatted_response = chain.invoke({
            "question": state["question"],
            "sql_query": state["sql_query"],
            "raw_results": raw_results_str,
            "chat_history": chat_history_str
        })
        state["formatted_response"] = formatted_response.content.strip()
        return state
        
    except Exception as e:
        state["error_message"] = f"Error formatting results: {str(e)}"
        return state
    

def build_graph(prompt_manager: PromptManager, db_connection: DatabaseConnection, llm) -> StateGraph:

    workflow = StateGraph(NL2SQLState)

    def _analyze_schema(state: NL2SQLState) -> NL2SQLState:
        if "chat_history" not in state:
            state["chat_history"] = []
        return analyze_schema(state, db_connection)

    def _find_relevant_tables(state: NL2SQLState) -> NL2SQLState:
        return select_relevant_tables(state, llm, prompt_manager)

    def _generate_sql(state: NL2SQLState) -> NL2SQLState:
        return generate_sql(state, llm, prompt_manager)

    def _execute_query(state: NL2SQLState) -> NL2SQLState:
        return execute_query(state, db_connection)

    def _format_results(state: NL2SQLState) -> NL2SQLState:
        state = format_results(state, llm, prompt_manager)
        # Update memory 
        if "chat_history" not in state:
            state["chat_history"] = []
        
        if not state.get("error_message") and state.get("formatted_response"):
            state["chat_history"].append({"role": "user", "content": state["question"]})
            state["chat_history"].append({"role": "assistant", "content": state["formatted_response"]})
        
        return state

    def _explain_query(state: NL2SQLState) -> NL2SQLState:
        return explain_query(state, llm, prompt_manager)

    # Add nodes to graph
    workflow.add_node("analyze_schema", _analyze_schema)
    workflow.add_node("find_relevant_tables", _find_relevant_tables)
    workflow.add_node("generate_sql", _generate_sql)
    workflow.add_node("execute_query", _execute_query)
    workflow.add_node("format_results", _format_results)
    workflow.add_node("explain_query", _explain_query)

    # Set entrypoint
    workflow.set_entry_point("analyze_schema")

    # Define edges
    workflow.add_edge(START, "analyze_schema")
    # workflow.add_edge("analyze_schema", "generate_sql")
    workflow.add_edge("analyze_schema", "find_relevant_tables")
    workflow.add_edge("find_relevant_tables", "generate_sql")
    workflow.add_edge("generate_sql", "execute_query")
    workflow.add_edge("execute_query", "format_results")
    workflow.add_edge("format_results", "explain_query")
    workflow.add_edge("explain_query", END)
    
    # Add memory persistence with checkpointer
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)