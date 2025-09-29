# backend/app/database.py
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from decimal import Decimal
from datetime import datetime
import json
from typing import List, Dict, Any

DB_USER = "test"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_database_url(db_name: str) -> str:
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}"

def get_engine(db_name: str):
    return create_engine(get_database_url(db_name))

def get_db_schema(engine) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns full table schema including:
    name, type, nullable, default
    """
    inspector = inspect(engine)
    schema_info = {}

    for table_name in inspector.get_table_names():
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),      # capture SQL type
                "nullable": col.get("nullable", True),
                "default": col.get("default")  # sometimes None
            })
        schema_info[table_name] = columns

    return schema_info

def json_serial(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def execute_sql(engine: Engine, sql: str):
    # Auto-commit using a transaction context
    with engine.begin() as conn:  # engine.begin() starts a transaction and commits automatically
        result = conn.execute(text(sql))

        # SELECT query → fetch results
        if sql.strip().lower().startswith("select"):
            rows = [dict(r._mapping) for r in result.fetchall()]
            return json.loads(json.dumps(rows, default=str))  # use str for json_serial fallback

        # DML query → return affected row count
        return result.rowcount

def init_query_history_table(engine: Engine):
    query = """
    CREATE TABLE IF NOT EXISTS query_history (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        db_name TEXT NOT NULL,
        user_prompt TEXT NOT NULL,
        generated_sql TEXT NOT NULL,
        result JSONB,
        created_at TIMESTAMP DEFAULT now()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(query))

def log_query_history(engine: Engine, user_id: str, db_name: str, user_prompt: str, generated_sql: str, result):
    result_json = json.dumps(result, default=json_serial)
    query = """
    INSERT INTO query_history (user_id, db_name, user_prompt, generated_sql, result, created_at)
    VALUES (:user_id, :db_name, :user_prompt, :generated_sql, CAST(:result AS JSONB), now())
    RETURNING id, created_at;
    """
    params = {
        "user_id": user_id,
        "db_name": db_name,
        "user_prompt": user_prompt,
        "generated_sql": generated_sql,
        "result": result_json
    }
    with engine.begin() as conn:
        row = conn.execute(text(query), params).fetchone()
    return {"id": row["id"], "created_at": row["created_at"]}

def fetch_history(engine: Engine, user_id: str, limit: int = 20):
    query = text(
        "SELECT id, user_prompt, generated_sql, result, created_at "
        "FROM query_history WHERE user_id = :user_id ORDER BY created_at DESC LIMIT :limit"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"user_id": user_id, "limit": limit}).fetchall()
        safe_rows = json.loads(json.dumps([dict(r._mapping) for r in rows], default=json_serial))
    return safe_rows
