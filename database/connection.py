import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_db")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager for PostgreSQL database connections."""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_cursor():
    """Context manager for database cursors returning results as dicts."""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor
            conn.commit()

def initialize_database():
    """Reads schema.sql and applies it to initialize tables and indexes."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        logger.error(f"Schema file not found at: {schema_path}")
        return

    logger.info("Initializing database schema...")
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
            conn.commit()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e

if __name__ == "__main__":
    initialize_database()
