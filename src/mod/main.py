import os
from dotenv import load_dotenv
from graph_builder import build_graph, NL2SQLState
from prompts import PromptManager
from db_connect import DatabaseConnection
from llm import setup_llm

load_dotenv()

def main():
    # Initialize components
    together_api_key = os.getenv('TOGETHER_API_KEY')
    db_type = os.getenv('DB_TYPE')
    db_host = os.getenv('DB_HOST')
    db_port = int(os.getenv('DB_PORT')) if os.getenv('DB_PORT') else None
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    # Setup LLM
    llm = setup_llm(together_api_key)
    
    # Setup database connection
    db_connection = DatabaseConnection()
    db_connection.connect_to_database(db_type, db_host, db_port, db_name, db_user, db_password)
    
    # Test the database connection
    if not db_connection.test_connection():
        print("Failed to connect to the database. Please check your credentials.")
        return
    
    # Setup prompt manager
    prompt_manager = PromptManager()
    
    # Build the graph
    workflow = build_graph(prompt_manager, db_connection, llm)
    
    # Get user question
    user_question = input("Enter your question: ")

    # Test with a sample question
    initial_state = NL2SQLState(
        question=user_question,
        db_schema="",
        relevant_tables=[],
        sql_query="",
        query_results=[],
        formatted_response="",
        explanation="",
        error_message="",
        chat_history=[]
    )
    
    # Execute the workflow
    try:
        # Use a consistent thread_id to maintain conversation history
        thread_config = {"configurable": {"thread_id": "user_session_1"}}
        
        result = workflow.invoke(initial_state, config=thread_config)

        print(f"Question: {result['question']}")
        print(f"SQL Query: {result['sql_query']}")
        print(f"Response: {result['formatted_response']}")
        print(f"Explanation: {result['explanation']}")
        
        if result.get('error_message'):
            print(f"Error: {result['error_message']}")
            
    except Exception as e:
        print(f"Execution failed: {str(e)}")

if __name__ == "__main__":
    main()