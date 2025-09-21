# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .database import get_engine, get_db_schema, execute_sql, init_query_history_table, log_query_history, fetch_history
from .langchain_nl2sql import create_sql_chain, run_sql_chain, memory  # import run_sql_chain

app = FastAPI(title="NL2SQL with LangChain + Gemini + History")

SESSION_HISTORY = {}  # {user_id: [ {prompt, sql, results, ts}, ... ]}

class QueryRequest(BaseModel):
    query: str
    db_name: str
    user_id: Optional[str] = "anonymous"

def build_schema_text(engine):
    schema = get_db_schema(engine)
    lines = []
    for t, cols in schema.items():
        lines.append(f"{t}({', '.join(cols)})")
    return "\n".join(lines)

def clean_sql(sql: str) -> str:
    """Remove any Markdown code fences from LLM output."""
    sql = sql.strip()
    if sql.startswith("```") and sql.endswith("```"):
        sql = "\n".join(sql.split("\n")[1:-1])
    return sql.strip()

@app.get("/")
def root():
    return {"message": "NL2SQL API (LangChain + Gemini + History)"}

@app.post("/query")
def run_query(req: QueryRequest):
    try:
        engine = get_engine(req.db_name)
        init_query_history_table(engine)
        schema_text = build_schema_text(engine)

        # Use our refined chain
        chain = create_sql_chain(schema_text)

        print("MEMORY DUMP:", memory)

        # Run the chain with refinement
        raw_sql = run_sql_chain(chain, schema_text, req.query)
        generated_sql = clean_sql(raw_sql)

        print("Generated SQL:", generated_sql)

        # Execute SQL
        results = execute_sql(engine, generated_sql)

        # Log history
        hist_row = log_query_history(
            engine, req.user_id, req.db_name,
            req.query, generated_sql, results
        )

        # Keep session history in memory
        SESSION_HISTORY.setdefault(req.user_id, []).append({
            "id": hist_row["id"],
            "prompt": req.query,
            "sql": generated_sql,
            "results": results
        })
        if len(SESSION_HISTORY[req.user_id]) > 200:
            SESSION_HISTORY[req.user_id] = SESSION_HISTORY[req.user_id][-200:]

        return {
            "sql": generated_sql,
            "results": results,
            "history_id": hist_row["id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
def get_history_api(user_id: str, limit: int = 20, db_name: Optional[str] = None):
    try:
        db_name = db_name or "imdb_movies"
        engine = get_engine(db_name)
        init_query_history_table(engine)
        rows = fetch_history(engine, user_id, limit=limit)
        return {"history": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
