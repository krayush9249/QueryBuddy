import re
import pandas as pd
from typing import Dict, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, validator, root_validator
from sqlalchemy import text
from langchain.chains import LLMChain
from langchain_together import TogetherLLM
from langgraph.graph import StateGraph, START, END
from db_connect import DatabaseConnection
from prompts import PromptManager


class NL2SQLState(TypedDict):
    """State for the NL2SQL workflow"""
    question: str
    db_schema: str
    relevant_tables: List[str]
    sql_query: str
    query_results: List[Dict]
    formatted_response: str
    explanation: str
    error_message: str
    # context: str

def build_workflow_graph() -> StateGraph:
    """Setup the LangGraph workflow for NL2SQL processing"""
    workflow = StateGraph(NL2SQLState)
    
    # Add workflow nodes
    workflow.add_node("analyze_schema", analyze_schema)
    workflow.add_node("select_relevant_tables", select_relevant_tables)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("format_results", format_results)
    workflow.add_node("explain_query", explain_query)
    
    # Define workflow edges
    workflow.add_edge(START, "analyze_schema")
    workflow.add_edge("analyze_schema", "select_relevant_tables")
    workflow.add_edge("select_relevant_tables", "generate_sql")
    workflow.add_edge("generate_sql", "execute_query")
    workflow.add_edge("execute_query", "format_results")
    workflow.add_edge("format_results", "explain_query")
    workflow.add_edge("explain_query", END)
    
    return workflow.compile()

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
        ..., description="List of relevant table names. Leave empty if no tables are found."
    )

def select_relevant_tables(state: NL2SQLState,
                           llm: TogetherLLM,
                           prompt_manager: PromptManager) -> NL2SQLState:
    """Select relevant tables based on the question using structured output"""
    try:
        prompt = prompt_manager.get_prompt('table_selection')
        structured_llm = llm.with_structured_output(TableSelectionOutput)
        chain = LLMChain(llm=structured_llm, prompt=prompt)

        response: TableSelectionOutput = chain.invoke({
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

    @validator('sql_query', pre=True)
    def clean_markdown(cls, v):
        v = re.sub(r'```sql\n?', '', v)
        v = re.sub(r'```\n?', '', v)
        v = re.sub(r'^sql\s*', '', v, flags=re.IGNORECASE)
        return v.strip()

    @validator('sql_query')
    def must_be_select_query(cls, v):
        if not v.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        return v

    @root_validator
    def check_prohibited_keywords(cls, values):
        sql = values.get("sql_query", "").upper()
        prohibited_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
        ]
        for keyword in prohibited_keywords:
            if keyword in sql:
                raise ValueError(f"Prohibited SQL operation detected: {keyword}")
        return values

def generate_sql(state: NL2SQLState, 
                llm: TogetherLLM,
                prompt_manager: PromptManager) -> NL2SQLState:
    """Generate SQL query based on the question and relevant tables"""
    try:
        prompt = prompt_manager.get_prompt('sql_generation')
        structured_llm = llm.with_structured_output(SQLQueryValidator)
        chain = LLMChain(llm=structured_llm, prompt=prompt)
        
        response = chain.run(
            question=state["question"],
            schema=state["db_schema"],
            tables=", ".join(state["relevant_tables"])
            # , context=state["context"]
        )
        state["sql_query"] = response.sql_query
        return state
   
    except Exception as e:
        state["error_message"] = f"Error generating SQL: {str(e)}"
        return state


def explain_query(state: NL2SQLState, 
                 llm: TogetherLLM,
                 prompt_manager: PromptManager) -> NL2SQLState:
    """Generate explanation for the SQL query"""
    if state.get("error_message"):
        return state  
    
    try:
        prompt = prompt_manager.get_prompt('query_explanation')
        chain = LLMChain(llm=llm, prompt=prompt)
        
        explanation = chain.run(
            question=state["question"],
            sql_query=state["sql_query"],
            schema=state["db_schema"]
        )
        
        state["explanation"] = explanation.strip()
        return state
    except Exception as e:
        state["error_message"] = f"Error generating explanation: {str(e)}"
        return state
    
    
def execute_query(state: NL2SQLState, 
                  db_connection: DatabaseConnection,
                  llm: TogetherLLM,
                  prompt_manager: PromptManager) -> NL2SQLState:
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
                  llm: TogetherLLM,
                  prompt_manager: PromptManager) -> NL2SQLState:
    """Format query results for display"""
    if state.get("error_message"):
        return state 
    
    if not state["query_results"]:
        state["formatted_response"] = "No results found for your query."
        return state
    
    try:
        prompt = prompt_manager.get_prompt('result_formatting')
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Convert results to string for prompt
        df = pd.DataFrame(state["query_results"])
        raw_results_str = df.to_string(index=False, max_rows=25)
        
        formatted_response = chain.run(
            question=state["question"],
            sql_query=state["sql_query"],
            raw_results=raw_results_str
        )

        state["formatted_response"] = formatted_response 
        return state
        
    except Exception as e:
        state["error_message"] = f"Error formatting results: {str(e)}"
        return state



