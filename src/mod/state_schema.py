from typing import List, Dict, TypedDict

# Define state with memory
class NL2SQLState(TypedDict):
    question: str
    sql_dialect: str
    db_schema: str
    relevant_tables: List[str]
    sql_query: str
    query_results: List[Dict]
    formatted_response: str
    explanation: str
    error_message: str
    chat_history: List[Dict[str, str]]