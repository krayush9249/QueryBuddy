from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from graph_utils import analyze_schema, select_relevant_tables, generate_sql, execute_query, format_results, explain_query
from mod.prompt import PromptManager
from db_connect import DatabaseConnection
from llm import TogetherLLM


# Define state with memory
class NL2SQLState(TypedDict):
    question: str
    db_schema: str
    relevant_tables: List[str]
    sql_query: str
    query_results: List[Dict]
    formatted_response: str
    explanation: str
    error_message: str
    chat_history: List[Dict[str, str]]  


def build_graph(prompt_manager: PromptManager, db_connection: DatabaseConnection, llm: TogetherLLM) -> StateGraph:

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
        return execute_query(state, db_connection, llm, prompt_manager)

    def _format_results(state: NL2SQLState) -> NL2SQLState:
        state = format_results(state, llm, prompt_manager)
        # Update memory
        if "chat_history" not in state:
            state["chat_history"] = []
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
    workflow.add_edge("analyze_schema", "find_relevant_tables")
    workflow.add_edge("find_relevant_tables", "generate_sql")
    workflow.add_edge("generate_sql", "execute_query")
    workflow.add_edge("execute_query", "format_results")
    workflow.add_edge("format_results", "explain_query")
    workflow.add_edge("explain_query", END)
    
    return workflow.compile()