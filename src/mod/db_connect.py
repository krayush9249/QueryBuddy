from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.utilities.sql_database import SQLDatabase


class DatabaseConnection:
    """Handles database connections for multiple database types"""

    def __init__(self):
        self.db = None
        self.engine = None

    def connect_to_database(self,
                            db_type: str,
                            db_host: str = "",
                            db_port: int = None,
                            db_name: str = "",
                            db_user: str = "",
                            db_password: str = "") -> SQLDatabase:
        """
        Connect to a database using SQLAlchemy and return a LangChain-compatible SQLDatabase object.
        Supported database types: MySQL, PostgreSQL, SQLite, SQL Server.
        """

        dialect_map = {
            "mysql": "mysql+pymysql",
            "postgresql": "postgresql+psycopg2",
            "sqlite": "sqlite",
            "mssql": "mssql+pyodbc"
        }

        if db_type.lower() not in dialect_map:
            raise ValueError(f"Unsupported database type: {db_type}. Supported types: {list(dialect_map.keys())}")

        try:
            if db_type.lower() == "sqlite":
                if not db_name:
                    raise ValueError("For SQLite, db_name must be the path to the .db file")
                uri = f"sqlite:///{db_name}"
            else:
                port_part = f":{db_port}" if db_port else ""
                uri = f"{dialect_map[db_type.lower()]}://{db_user}:{db_password}@{db_host}{port_part}/{db_name}"

            self.engine = create_engine(uri, echo=False)
            self.db = SQLDatabase.from_engine(self.engine)
            return self.db

        except SQLAlchemyError as e:
            raise ConnectionError(f"Failed to connect to {db_type} database: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error connecting to database: {e}")

    def get_schema_info(self) -> str:
        if not self.db:
            raise RuntimeError("No database connection established")
        try:
            return self.db.get_table_info()
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve schema info: {e}")

    def get_table_names(self) -> list:
        if not self.db:
            raise RuntimeError("No database connection established")
        try:
            return self.db.get_usable_table_names()
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve table names: {e}")

    def test_connection(self) -> bool:
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
        
    def close_connection(self):
        """Close the SQLAlchemy engine to release the DB connection"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.db = None