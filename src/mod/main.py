import os
from dotenv import load_dotenv
from graph import build_graph
from state_schema import NL2SQLState
from prompts import PromptManager
from db_connect import DatabaseConnection
from llm import LLMSetup

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
    llm = LLMSetup(together_api_key)
    
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
    
    # Use a consistent thread_id to maintain conversation history
    thread_config = {"configurable": {"thread_id": "user_session_1"}}
    
    print("Welcome to NL2SQL Chat! Type 'quit' or 'exit' to end the session.")
    print("-" * 100)
    
    # Main conversation loop
    while True:
        # Get user question
        # user_question = input("\nEnter your question: ").strip()
        user_question = "Tell me the number of employees hired after the year 1999"
        
        # Check for exit conditions
        if user_question.lower() in ['quit', 'exit']:
            print("Thank you for using NL2SQL Chat. Goodbye!")
            break
        
        # Skip empty questions
        if not user_question:
            print("Please enter a valid question.")
            continue
        
        # Create state for current question
        current_state = NL2SQLState(
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
            result = workflow.invoke(current_state, config=thread_config)
            
            print(f"\nDB Schema: {result['db_schema']}")
            print(f"\nRelevant Tables: {result['relevant_tables']}")
            print(f"\nSQL Query: {result['sql_query']}")
            print(f"\nResponse: {result['formatted_response']}")
            print(f"\nExplanation: {result['explanation']}")
            
            if result.get('error_message'):
                print(f"\nError: {result['error_message']}")
                
        except Exception as e:
            print(f"\nExecution failed: {str(e)}")
        
        print("-" * 100)

if __name__ == "__main__":
    main()