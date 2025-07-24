from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.utilities.sql_database import SQLDatabase
from typing import Tuple, Optional, Dict, List, Any

def build_connection_uri(db_type: str, 
                        db_host: str = "",
                        db_port: Optional[int] = None,
                        db_name: str = "",
                        db_user: str = "",
                        db_password: str = "") -> str:
    """
    Build database connection URI based on database type and parameters.
    """

    DIALECT_MAP = {
    "mysql": "mysql+pymysql",
    "postgresql": "postgresql+psycopg2",
    "sqlite": "sqlite",
    "mssql": "mssql+pyodbc"
    }
    
    if db_type.lower() not in DIALECT_MAP:
        raise ValueError(f"Unsupported database type: {db_type}. Supported types: {list(DIALECT_MAP.keys())}")
    
    if db_type.lower() == "sqlite":
        return f"sqlite:///{db_name}"  # db_name is full path to .db file
    else:
        port_part = f":{db_port}" if db_port else ""
        return f"{DIALECT_MAP[db_type.lower()]}://{db_user}:{db_password}@{db_host}{port_part}/{db_name}"


def create_engine_connection(connection_uri: str) -> Any:
    """
    Create SQLAlchemy engine from connection URI.
    """
    try:
        return create_engine(connection_uri, echo=False)
    except SQLAlchemyError as e:
        raise ConnectionError(f"Failed to create database engine: {e}")
    except Exception as e:
        raise ConnectionError(f"Unexpected error creating engine: {e}")


def create_langchain_database(engine: Any) -> SQLDatabase:
    """
    Create LangChain-compatible SQLDatabase object from SQLAlchemy engine.
    """
    try:
        return SQLDatabase.from_engine(engine)
    except Exception as e:
        raise ConnectionError(f"Failed to create LangChain SQLDatabase: {e}")


def connect_to_database(db_type: str,
                       db_host: str = "",
                       db_port: Optional[int] = None,
                       db_name: str = "",
                       db_user: str = "",
                       db_password: str = "") -> Tuple[SQLDatabase, Any]:
    """
    Connect to a database and return both LangChain SQLDatabase and SQLAlchemy engine.
    """
    connection_uri = build_connection_uri(db_type, db_host, db_port, db_name, db_user, db_password)
    engine = create_engine_connection(connection_uri)
    db = create_langchain_database(engine)
    
    return db, engine


def get_schema_info(db: SQLDatabase) -> str:
    """
    Get database schema information.
    """
    try:
        return db.get_table_info()
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve schema info: {e}")


def get_table_names(db: SQLDatabase) -> List[str]:
    """
    Get list of table names in the database.
    """
    try:
        return db.get_usable_table_names()
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve table names: {e}")


def test_connection(engine: Any) -> bool:
    """
    Test if the database connection is active.
    """
    if not engine:
        return False
    
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except:
        return False


def validate_database_config(config: Dict[str, Any]) -> None:
    """
    Validate database configuration dictionary.
    """
    required_keys = ['db_type', 'db_name']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

def get_database_config_template(db_type: str) -> Dict[str, Any]:
    """
    Get a configuration template for different database types.
    """
    templates = {
        'sqlite': {
            'db_type': 'sqlite',
            'db_name': 'path/to/database.db'
        },
        'mysql': {
            'db_type': 'mysql',
            'db_host': 'localhost',
            'db_port': 3306,
            'db_name': 'your_database',
            'db_user': 'your_username',
            'db_password': 'your_password'
        },
        'postgresql': {
            'db_type': 'postgresql',
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'your_database',
            'db_user': 'your_username',
            'db_password': 'your_password'
        },
        'mssql': {
            'db_type': 'mssql',
            'db_host': 'localhost',
            'db_port': 1433,
            'db_name': 'your_database',
            'db_user': 'your_username',
            'db_password': 'your_password'
        }
    }
    
    if db_type.lower() not in templates:
        raise ValueError(f"Unknown database type: {db_type}")
    
    return templates[db_type.lower()]

def get_database_info(db: SQLDatabase) -> Dict[str, Any]:
    """
    Get comprehensive database information.
    """
    return {
        'schema_info': get_schema_info(db),
        'table_names': get_table_names(db)
    }