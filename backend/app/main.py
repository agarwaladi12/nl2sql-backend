# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .database import get_engine, get_db_schema, execute_sql, init_query_history_table, log_query_history, fetch_history
from .langchain_nl2sql import create_sql_chain, run_sql_chain, memory  # updated imports
from .dml_validator import validate_and_cast_dml, normalize_schema
import json

app = FastAPI(title="NL2SQL with LangChain + Gemini + History")


class QueryRequest(BaseModel):
    query: str
    db_name: str
    user_id: Optional[str] = "anonymous"

class ConfirmRequest(BaseModel):
    user_id: str
    db_name: str
    sql: str
    confirm: bool

def build_schema_text(schema_map):
    lines = []
    for table, cols in schema_map.items():
        col_desc = []
        for cname, meta in cols.items():
            req_flag = "required" if meta["required"] else "nullable"
            typ = meta.get("type") or "UNKNOWN"
            col_desc.append(f"{cname}({typ}, {req_flag})")
        lines.append(f"{table}: {', '.join(col_desc)}")
    return "\n".join(lines)

def clean_sql(sql: str) -> str:
    """Remove any Markdown code fences from LLM output."""
    sql = sql.strip()
    if sql.startswith("```") and sql.endswith("```"):
        sql = "\n".join(sql.split("\n")[1:-1])
    return sql.strip()

def clean_llm_output(llm_text: str) -> str:
    """
    Remove Markdown code fences and extra whitespace from LLM output.
    """
    if not isinstance(llm_text, str):
        return llm_text
    llm_text = llm_text.strip()
    if llm_text.startswith("```") and llm_text.endswith("```"):
        # Remove first and last fence lines
        lines = llm_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        llm_text = "\n".join(lines)
    return llm_text.strip()

@app.get("/")
def root():
    return {"message": "NL2SQL API (LangChain + Gemini + History)"}

@app.post("/query")
def run_query(req: QueryRequest):
    try:
        engine = get_engine(req.db_name)
        init_query_history_table(engine)
        raw_schema = get_db_schema(engine)
        schema_map = normalize_schema(raw_schema)
        schema_text = build_schema_text(schema_map)

        # Create SQL chain
        chain = create_sql_chain(schema_text)

        # Run the chain (LLM may return JSON with SQL + suggestions)
        result = run_sql_chain(chain, schema_text, req.query, req.user_id)

        if result.get("clarification_required"):
            return {"message": result["message"], "clarification_required": True}
        
        # Run the chain (LLM may return JSON with SQL + suggestions)
        llm_output = result  # raw LLM response

        # Case 1: If "sql" field itself contains JSON (like your example)
        if isinstance(llm_output, dict) and isinstance(llm_output.get("sql"), str):
            inner_sql = clean_llm_output(llm_output["sql"])
            try:
                parsed_inner = json.loads(inner_sql)
                # Replace "sql" with real SQL
                llm_output["sql"] = parsed_inner.get("sql", "").strip()
                llm_output["suggestions"] = parsed_inner.get("suggestions", [])
            except Exception:
                llm_output["sql"] = inner_sql.strip()

        generated_sql = llm_output.get("sql", "").strip()
        suggestions = llm_output.get("suggestions", [])
        requires_confirmation = llm_output.get("requires_confirmation", False)

        print("Generated SQL:", generated_sql)
        print("Suggestions:", suggestions)

        # Detect DML queries
        sql_upper = generated_sql.strip().upper()
        is_dml = sql_upper.startswith(("INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT"))
        session_history = memory[req.user_id]["history"]

        if is_dml or requires_confirmation:
            session_history.append({
                "query": req.query,
                "sql": generated_sql,
                "suggestions": suggestions,
                "requires_confirmation": is_dml,
                "executed": False
            })
            return {
                "clarification_required": False,
                "sql": generated_sql,
                "suggestions": suggestions,
                "message": llm_output.get(
                    "message",
                    "This is a DML query. Do you want to execute it? Reply with CONFIRM_EXECUTE to proceed."
                ),
                "requires_confirmation": True
            }

        # Execute SELECT normally
        results = execute_sql(engine, generated_sql)

        # Log history
        hist_row = log_query_history(
            engine, req.user_id, req.db_name,
            req.query, generated_sql, results
        )

        session_history.append({
            "query": req.query,
            "sql": generated_sql,
            "suggestions": suggestions,
            "requires_confirmation": is_dml,
            "executed": True
        })
        if len(session_history) > memory[req.user_id]["max_history"]:
            memory[req.user_id]["history"] = session_history[-memory[req.user_id]["max_history"]:]

        return {
            "clarification_required": False,
            "sql": generated_sql,
            "results": results,
            "suggestions": suggestions,
            "history_id": hist_row["id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm_dml")
def confirm_dml(req: ConfirmRequest):
    try:
        # Make sure session exists
        if req.user_id not in memory:
            raise HTTPException(status_code=400, detail="No session history found for this user.")

        session_history = memory[req.user_id]["history"]

        # Find the query in history
        matching = next((item for item in session_history if item["sql"] == req.sql), None)
        if not matching:
            raise HTTPException(status_code=400, detail="SQL not found in session history.")
        if not matching.get("requires_confirmation", False):
            raise HTTPException(status_code=400, detail="This query does not require confirmation.")
        if matching.get("executed", False):
            raise HTTPException(status_code=400, detail="This query has already been executed.")

        # Handle user cancel
        if not req.confirm:
            return {"message": "DML query execution cancelled by user."}

        # Validate before execution
        engine = get_engine(req.db_name)
        raw_schema = get_db_schema(engine)
        schema_map = normalize_schema(raw_schema)
        validation = validate_and_cast_dml(req.sql, schema_map)
        if not validation["valid"]:
            return {"message": validation["message"]}

        # Execute if valid
        affected_rows = execute_sql(engine, validation["sql"])

        # Mark query as executed in session history
        matching["executed"] = True

        return {"message": "DML executed successfully", "affected_rows": affected_rows}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}/{db_name}")
def get_history_api(user_id: str, limit: int = 20, db_name: Optional[str] = None):
    try:
        engine = get_engine(db_name)
        init_query_history_table(engine)
        rows = fetch_history(engine, user_id, limit=limit)
        return {"history": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
