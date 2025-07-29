import streamlit as st
import os
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph import build_graph
from state_schema import NL2SQLState
from prompts import PromptManager
from db_connect import DatabaseConnection
from llms import setup_groq_llm, setup_together_llm

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="QueryBuddy - SQL Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .connection-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    
    .connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .disconnected {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .query-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4f46e5;
        margin: 1rem 0;
    }
    
    .sql-code {
        background-color: #1e1e1e;
        color: #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        overflow-x: auto;
    }
    
    .results-header {
        background-color: #4f46e5;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
        font-weight: bold;
    }
    
    .history-item {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        margin-bottom: 1rem;
    }
    
    .error-message {
        background-color: #fef2f2;
        color: #dc2626;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fecaca;
    }
    
    .success-message {
        background-color: #f0fdf4;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
    }
    
    .center-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None
if 'workflow' not in st.session_state:
    st.session_state.workflow = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'thread_config' not in st.session_state:
    st.session_state.thread_config = {"configurable": {"thread_id": "streamlit_session"}}

def initialize_components():
    """Initialize LLM and other components based on user selection."""
    try:
        # Step 1: Choose provider
        provider = st.selectbox(
            "Choose API Provider",
            ["Groq", "Together AI"],
            index=0,
            key="api_provider_select"
        )
        # Step 2: Set model and prompt for API key
        if provider == "Groq":
            model = st.selectbox(
                "Choose Groq Model",
                ["meta-llama/llama-4-scout-17b-16e-instruct", 
                 "meta-llama/llama-4-maverick-17b-128e-instruct"],
                key="groq_model_select"
            )
            groq_api_key = st.text_input("Enter Groq API Key", type="password", key="groq_api_key", value=os.getenv("GROQ_API_KEY", ""))
            if not groq_api_key:
                st.warning("Please enter your Groq API key.")
                return None
            llm = setup_groq_llm(groq_api_key, model)
        else:  # Together AI
            model = st.selectbox(
                "Choose Together AI Model",
                ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                 "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"],
                key="together_model_select"
            )
            together_api_key = st.text_input("Enter Together AI Key", type="password", key="together_api_key", value=os.getenv("TOGETHER_API_KEY", ""))
            if not together_api_key:
                st.warning("Please enter your Together AI API key.")
                return None
            llm = setup_together_llm(together_api_key, model)
        # Setup prompt manager
        prompt_manager = PromptManager()
        return llm, prompt_manager
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None

def connect_to_database(db_config: Dict[str, Any], llm, prompt_manager) -> bool:
    """Connect to database with given configuration"""
    try:
        db_connection = DatabaseConnection()
        db_connection.connect_to_database(
            db_type=db_config['type'],
            db_host=db_config['host'],
            db_port=db_config['port'],
            db_name=db_config['name'],
            db_user=db_config['user'],
            db_password=db_config['password']
        )
        
        if db_connection.test_connection():
            st.session_state.db_connection = db_connection
            st.session_state.connected = True
            
            # Initialize workflow with provided components
            if llm and prompt_manager:
                st.session_state.workflow = build_graph(prompt_manager, db_connection, llm)
                return True
        return False
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return False

def process_query(question: str) -> Dict[str, Any]:
    """Process natural language query using the workflow"""
    if not st.session_state.workflow or not st.session_state.connected:
        return {"error": "No database connection or workflow initialized"}
    
    try:
        # Create state for current question
        current_state = NL2SQLState(
            question=question,
            sql_dialect=st.session_state.db_config['type'],
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
        with st.spinner("QueryBuddy is thinking..."):
            result = st.session_state.workflow.invoke(current_state, config=st.session_state.thread_config)
        
        return result
    except Exception as e:
        return {"error": f"Query processing failed: {str(e)}"}

def display_results(result: Dict[str, Any]):
    """Display query results in tabs"""
    if result.get('error'):
        st.markdown(f'<div class="error-message">❌ {result["error"]}</div>', unsafe_allow_html=True)
        return
    
    if result.get('error_message'):
        st.markdown(f'<div class="error-message">❌ {result["error_message"]}</div>', unsafe_allow_html=True)
        return
    
    # Create tabs for different views 
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Response", "📊 Results", "💻 SQL Query", "📝 Explanation"])
    
    with tab1:
        if result.get('formatted_response'):
            st.markdown("**AI Response:**")
            st.write(result['formatted_response'])
        else:
            st.info("No AI response available.")
    
    with tab2:
        if result.get('query_results'):
            st.markdown('<div class="results-header">Query Results</div>', unsafe_allow_html=True)
            df = pd.DataFrame(result['query_results'])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Showing {len(df)} rows")
        else:
            st.info("No results found for your query.")
    
    with tab3:
        if result.get('sql_query'):
            st.markdown("**Generated SQL Query:**")
            st.markdown(f'<div class="sql-code">{result["sql_query"]}</div>', unsafe_allow_html=True)
            
            # Copy button for SQL
            if st.button("Copy SQL", key="copy_sql"):
                st.code(result['sql_query'], language='sql')
        else:
            st.info("No SQL query generated.")
    
    with tab4:
        if result.get('explanation'):
            st.markdown("**Query Explanation:**")
            st.write(result['explanation'])
        else:
            st.info("No explanation available.")

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 QueryBuddy</h1>
        <p>Your AI-powered SQL Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        # API Key Configuration 
        with st.expander("API Configuration", expanded=True):
            components = initialize_components()
        
        st.divider()
        
        # Database Configuration
        with st.expander("Database Connection", expanded=True):
            db_type = st.selectbox(
                "Database Type",
                ["mysql", "postgresql", "sqlite", "mssql"],
                help="Select your database type",
                key="db_type_select"
            )
            
            if db_type == "sqlite":
                db_name = st.text_input("Database File Path", placeholder="/path/to/database.db", key="sqlite_db_path")
                db_host = db_port = db_user = db_password = ""
            else:
                col1, col2 = st.columns(2)
                with col1:
                    db_host = st.text_input("Host", key="db_host")
                with col2:
                    db_port = st.text_input("Port", key="db_port")
                
                db_name = st.text_input("Database Name", key="db_name")
                db_user = st.text_input("Username", key="db_user")
                db_password = st.text_input("Password", type="password", key="db_password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                connect_clicked = st.button(
                    "Connect", 
                    type="primary", 
                    disabled=st.session_state.connected,
                    key="db_connect_btn",
                    use_container_width=True
                )
            
            with col2:
                disconnect_clicked = st.button(
                    "Disconnect", 
                    disabled=not st.session_state.connected,
                    key="db_disconnect_btn",
                    use_container_width=True
                )
            
            # Handle connection
            if connect_clicked:
                if db_type == "sqlite" and not db_name:
                    st.error("Please provide database file path for SQLite")
                elif db_type != "sqlite" and not all([db_host, db_name, db_user]):
                    st.error("Please fill in all required fields")
                elif not components:
                    st.error("Please configure API settings first")
                else:
                    # Convert port to int 
                    port_value = None
                    if db_port:
                        try:
                            port_value = int(db_port)
                        except ValueError:
                            st.error("Port must be a valid number")
                            return
                    
                    db_config = {
                        'type': db_type,
                        'host': db_host,
                        'port': port_value,
                        'name': db_name,
                        'user': db_user,
                        'password': db_password
                    }
                    st.session_state.db_config = db_config
                    
                    llm, prompt_manager = components
                    if connect_to_database(db_config, llm, prompt_manager):
                        st.success("✅ Connected successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Connection failed!")
            
            # Handle disconnection
            if disconnect_clicked:
                st.session_state.connected = False
                st.session_state.db_connection = None
                st.session_state.workflow = None
                st.rerun()
        
        st.divider()
        
        # Connection status
        if st.session_state.connected:
            st.markdown('<div class="connection-status connected">🟢 Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-status disconnected">🔴 Disconnected</div>', unsafe_allow_html=True)
    
    # Main content area
    st.header("Ask Your Question")
    
    # Query input
    question = st.text_area(
        "Enter your question:",
        height=100,
        disabled=not st.session_state.connected,
        key="question_input"
    )
    
    # Center-aligned buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col2:
        enter_clicked = st.button("Enter", type="primary", disabled=not st.session_state.connected, key="enter_btn")
    
    with col4:
        clear_clicked = st.button("Clear", disabled=not st.session_state.connected, key="clear_btn")
    
    # Handle button clicks
    if enter_clicked:
        if question.strip():
            result = process_query(question.strip())
            
            # Add to history
            st.session_state.query_history.append({
                'timestamp': datetime.now(),
                'question': question.strip(),
                'result': result
            })
            
            # Display results
            display_results(result)
        else:
            st.warning("Please enter a question")
    
    if clear_clicked:
        st.rerun()
    
    # Query History Section
    st.divider()
    st.header("Query History")
    
    if st.session_state.query_history:
        if st.button("Clear History", key="clear_history_btn"):
            st.session_state.query_history = []
            st.rerun()
        
        # Show recent queries
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):  # Last 5 queries
            with st.expander(f"Query {len(st.session_state.query_history) - i}", expanded=False):
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
                
                if item['result'].get('sql_query'):
                    st.code(item['result']['sql_query'][:100] + "..." if len(item['result']['sql_query']) > 100 else item['result']['sql_query'], language='sql')
                
                if st.button(f"View Results", key=f"view_{i}"):
                    display_results(item['result'])
    else:
        st.info("No queries yet.")
    
    # Footer information
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 QueryBuddy - Powered by LangGraph, Groq, and Together AI</p>
        <p>⚠️ Only SELECT queries are supported for security reasons</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()