import os
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from db_connection import DatabaseConnection, create_database_connection, get_database_config_template
from sql_chain import SQLChain, SQLChainFactory
from prompts import PromptManager


class NL2SQLApplication:
    """Main application class for NL2SQL processing"""
    
    def __init__(self, together_api_key: str):
        """
        Initialize the NL2SQL application
        
        Args:
            together_api_key: API key for Together AI
        """
        self.together_api_key = together_api_key
        self.db_connection = None
        self.sql_chain = None
        self.session_active = False
        self.session_id = None
        
        # Initialize components
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the SQL processing chain"""
        try:
            self.sql_chain = SQLChainFactory.create_default_chain(self.together_api_key)
            print("✓ SQL processing chain initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize SQL chain: {e}")
            sys.exit(1)
    
    def connect_database(self, db_config: Dict[str, Any]) -> bool:
        """
        Connect to database using configuration
        
        Args:
            db_config: Database configuration dictionary
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.db_connection = create_database_connection(db_config)
            self.sql_chain.set_database_connection(self.db_connection)
            
            # Test connection
            if self.db_connection.test_connection():
                print(f"✓ Connected to {db_config['db_type']} database successfully")
                
                # Get basic database info
                table_names = self.db_connection.get_table_names()
                print(f"✓ Found {len(table_names)} tables: {', '.join(table_names[:5])}")
                if len(table_names) > 5:
                    print(f"   ... and {len(table_names) - 5} more tables")
                
                self.session_active = True
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                return True
            else:
                print("✗ Database connection test failed")
                return False
                
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def process_query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            question: Natural language question
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing query results
        """
        if not self.session_active:
            return {
                "success": False,
                "error": "No active database session. Please connect to a database first."
            }
        
        if verbose:
            print(f"\n🔍 Processing question: {question}")
            print("-" * 50)
        
        try:
            # Analyze question complexity
            complexity = self.sql_chain.analyze_question_complexity(question)
            if verbose:
                print(f"📊 Question complexity: {complexity['complexity_level']}")
            
            # Process the question
            result = self.sql_chain.process_question(question)
            
            if result["success"]:
                if verbose:
                    self._print_success_result(result)
                return result
            else:
                if verbose:
                    print(f"❌ Error: {result['error']}")
                return result
                
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "question": question
            }
            if verbose:
                print(f"❌ Unexpected error: {e}")
            return error_result
    
    def _print_success_result(self, result: Dict[str, Any]):
        """Print formatted successful query result"""
        print(f"✅ Query processed successfully!")
        print(f"\n📋 Generated SQL Query:")
        print(f"```sql\n{result['sql_query']}\n```")
        
        print(f"\n📊 Results ({result['results_count']} rows):")
        print(result['formatted_results'])
        
        print(f"\n💡 Explanation:")
        print(result['explanation'])
        
        if result.get('relevant_tables'):
            print(f"\n🗂️ Tables used: {', '.join(result['relevant_tables'])}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database connection"""
        if not self.db_connection:
            return {"status": "No database connected"}
        
        try:
            info = self.db_connection.get_connection_info()
            info.update({
                "table_count": len(self.db_connection.get_table_names()),
                "table_names": self.db_connection.get_table_names()
            })
            return info
        except Exception as e:
            return {"status": "Error", "error": str(e)}
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        if not self.sql_chain:
            return {"status": "No session active"}
        
        stats = self.sql_chain.get_session_stats()
        stats.update({
            "session_id": self.session_id,
            "session_active": self.session_active
        })
        return stats
    
    def get_conversation_history(self) -> list:
        """Get the conversation history for the current session"""
        if not self.sql_chain:
            return []
        return self.sql_chain.get_conversation_history()
    
    def reset_session(self):
        """Reset the current session"""
        if self.sql_chain:
            self.sql_chain.reset_session()
        
        if self.db_connection:
            self.db_connection.close_connection()
        
        self.db_connection = None
        self.session_active = False
        self.session_id = None
        print("🔄 Session reset successfully")
    
    def interactive_mode(self):
        """Run the application in interactive mode"""
        print("🚀 NL2SQL Interactive Mode")
        print("=" * 50)
        
        # Check if database is connected
        if not self.session_active:
            print("No database connection found. Let's set one up!")
            if not self._interactive_database_setup():
                print("Cannot proceed without database connection. Exiting.")
                return
        
        print("\n💬 You can now ask questions about your database!")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n🤔 Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                elif user_input.lower() == 'status':
                    self._print_status()
                    continue
                elif user_input.lower() == 'history':
                    self._print_history()
                    continue
                elif user_input.lower() == 'reset':
                    self.reset_session()
                    if not self._interactive_database_setup():
                        break
                    continue
                elif user_input.lower() == 'tables':
                    self._print_tables()
                    continue
                
                # Process the question
                result = self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Cleanup
        self.reset_session()
    
    def _interactive_database_setup(self) -> bool:
        """Interactive database setup"""
        print("\n🔧 Database Setup")
        print("Supported databases: mysql, postgresql, sqlite, mssql")
        
        db_type = input("Database type: ").strip().lower()
        
        if db_type not in ['mysql', 'postgresql', 'sqlite', 'mssql']:
            print("❌ Unsupported database type")
            return False
        
        try:
            template = get_database_config_template(db_type)
            config = {}
            
            for key, default_value in template.items():
                if key == 'db_password':
                    import getpass
                    value = getpass.getpass(f"{key} [{default_value}]: ") or default_value
                else:
                    value = input(f"{key} [{default_value}]: ") or default_value
                
                # Convert port to int if needed
                if key == 'db_port' and value:
                    try:
                        value = int(value)
                    except ValueError:
                        print(f"❌ Invalid port number: {value}")
                        return False
                
                config[key] = value
            
            return self.connect_database(config)
            
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return False
    
    def _print_help(self):
        """Print help information"""
        print("\n📚 Available Commands:")
        print("  help     - Show this help message")
        print("  status   - Show session and database status")
        print("  history  - Show conversation history")
        print("  tables   - Show available database tables")
        print("  reset    - Reset session and reconnect to database")
        print("  quit     - Exit the application")
        print("\n💡 Tips:")
        print("  - Ask natural language questions about your data")
        print("  - Be specific about what you want to know")
        print("  - The AI will generate SQL and explain the results")
    
    def _print_status(self):
        """Print current status"""
        print(f"\n📊 Session Status:")
        session_info = self.get_session_info()
        for key, value in session_info.items():
            print(f"  {key}: {value}")
        
        print(f"\n🗄️ Database Status:")
        db_info = self.get_database_info()
        for key, value in db_info.items():
            if key != 'table_names':  # Don't print all table names here
                print(f"  {key}: {value}")
    
    def _print_history(self):
        """Print conversation history"""
        history = self.get_conversation_history()
        if not history:
            print("\n📝 No conversation history yet")
            return
        
        print(f"\n📝 Conversation History ({len(history)} messages):")
        for i, msg in enumerate(history, 1):
            role = "👤 You" if msg['role'] == 'user' else "🤖 AI"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  {i}. {role}: {content}")
    
    def _print_tables(self):
        """Print available database tables"""
        if not self.db_connection:
            print("❌ No database connection")
            return
        
        try:
            tables = self.db_connection.get_table_names()
            print(f"\n🗂️ Available Tables ({len(tables)}):")
            for i, table in enumerate(tables, 1):
                print(f"  {i}. {table}")
        except Exception as e:
            print(f"❌ Error getting table names: {e}")


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load database configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return {}


def main():
    """Main entry point for the application"""
    print("🎯 NL2SQL Application")
    print("=" * 30)
    
    # Get Together API key
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        print("❌ TOGETHER_API_KEY environment variable not set")
        print("Please set your Together AI API key as an environment variable:")
        print("export TOGETHER_API_KEY='your_api_key_here'")
        sys.exit(1)
    
    # Create application instance
    try:
        app = NL2SQLApplication(together_api_key)
    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "interactive":
            app.interactive_mode()
        
        elif command == "config" and len(sys.argv) > 2:
            # Load database config from file
            config_path = sys.argv[2]
            config = load_config_from_file(config_path)
            if config and app.connect_database(config):
                print("Database connected successfully!")
                if len(sys.argv) > 3:
                    # Process question from command line
                    question = " ".join(sys.argv[3:])
                    result = app.process_query(question)
                else:
                    # Enter interactive mode
                    app.interactive_mode()
        
        elif command == "help":
            print("\nUsage:")
            print("  python main.py interactive              # Interactive mode")
            print("  python main.py config config.json      # Load config and start interactive")
            print("  python main.py config config.json 'question'  # Process single question")
            print("  python main.py help                     # Show this help")
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python main.py help' for usage information")
    
    else:
        # Default to interactive mode
        app.interactive_mode()


if __name__ == "__main__":
    main()