from typing import Dict, Any, List
from langchain_together import TogetherLLM
from langchain.memory import ConversationBufferMemory
from mod.graph import build_workflow_graph, NL2SQLState
from db_connect import DatabaseConnection
from prompts import PromptManager


question: str
db_connection: DatabaseConnection
llm: TogetherLLM
prompt_manager: PromptManager
memory: ConversationBufferMemory

# Initialize state
initial_state = NL2SQLState(
    question=question,
    db_schema="",
    relevant_tables=[],
    sql_query="",
    query_results=[],
    formatted_results="",
    explanation="",
    error_message="",
    context=""
)

def main():
    # Create the compiled workflow
    workflow = build_workflow_graph()
    
    # Execute workflow
    final_state = workflow.invoke(initial_state)
    
    print("Response:\n", final_state.get("formatted_response", "No response"))
    print("\nSQL Query:\n", final_state.get("sql_query", ""))
    print("\nExplanation:\n", final_state.get("explanation", ""))
    print("\nRaw Results (first 25 rows):")
    for row in final_state.get("query_results", [])[:25]:
        print(row)

if __name__ == "__main__":
    main()

