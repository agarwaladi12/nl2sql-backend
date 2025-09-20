#!/usr/bin/env python3
"""
build_schema_index.py

Usage (run from project root or backend folder):
PYTHONPATH=backend python3 -m app.scripts.build_schema_index --db nl2sql

It reads DB connection from app.database.get_engine(db_name).
Outputs: backend/app/data/schema_index_<db_name>.json
"""

import os
import json
import argparse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# ensure we can import your app.database
from app.database import get_engine  # path assumes package `app` is importable

SAMPLE_LIMIT = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_str(v):
    """Convert value to string for JSON safely"""
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return repr(v)

def build_index_for_db(db_name: str):
    engine = get_engine(db_name)
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
    except Exception as e:
        raise RuntimeError(f"Failed to inspect engine: {e}")

    schema_index = {"db_name": db_name, "tables": {}, "table_docs": [], "column_docs": []}

    tables = inspector.get_table_names()
    print(f"Found {len(tables)} tables in database '{db_name}'")

    for table in tables:
        try:
            cols = inspector.get_columns(table)
            pk = inspector.get_pk_constraint(table).get("constrained_columns", []) if inspector.get_pk_constraint(table) else []
            fks_raw = inspector.get_foreign_keys(table)
            fks = []
            for fk in fks_raw:
                referred_table = fk.get("referred_table")
                constrained = fk.get("constrained_columns", [])
                referred_cols = fk.get("referred_columns", [])
                fks.append({"constrained_columns": constrained, "referred_table": referred_table, "referred_columns": referred_cols})

            # ===== Convert columns to JSON-serializable format =====
            columns_clean = []
            for c in cols:
                columns_clean.append({
                    "name": c["name"],
                    "type": str(c.get("type")),       # stringify SQLAlchemy type
                    "nullable": c.get("nullable"),
                    "default": safe_str(c.get("default")),
                })

            schema_index["tables"][table] = {
                "columns": columns_clean,
                "primary_key": pk,
                "foreign_keys": fks
            }

            # sample values per column (limit SAMPLE_LIMIT)
            samples = {}
            with engine.connect() as conn:
                for c in [col["name"] for col in cols]:
                    try:
                        q = text(f'SELECT "{c}" FROM "{table}" WHERE "{c}" IS NOT NULL LIMIT :lim')
                        rows = conn.execute(q, {"lim": SAMPLE_LIMIT}).fetchall()
                        vals = [safe_str(r[0]) for r in rows]
                    except Exception:
                        vals = []
                    samples[c] = vals

            # Build table-level doc
            col_summaries = []
            for c in columns_clean:
                col_name = c["name"]
                dtype = c["type"]
                sample_vals = samples.get(col_name, [])[:5]
                sample_str = ", ".join([v for v in sample_vals if v is not None]) if sample_vals else ""
                col_summaries.append(f"{col_name} ({dtype})" + (f" — examples: {sample_str}" if sample_str else ""))

            pk_str = ", ".join(pk) if pk else "None"
            fk_str = "; ".join([f'{", ".join(fk["constrained_columns"])} -> {fk["referred_table"]}({", ".join(fk["referred_columns"])})' for fk in fks]) if fks else "None"

            table_doc_text = f"Table {table}: columns: " + "; ".join(col_summaries) + f". Primary key: {pk_str}. Foreign keys: {fk_str}."
            schema_index["table_docs"].append({"table": table, "text": table_doc_text})

            # column-level docs
            for c in columns_clean:
                col_name = c["name"]
                dtype = c["type"]
                samples_text = ", ".join([v for v in samples.get(col_name, []) if v is not None])[:200]
                col_doc_text = f"Column {table}.{col_name} ({dtype}). Examples: {samples_text}."
                schema_index["column_docs"].append({"table": table, "column": col_name, "text": col_doc_text})

            print(f"Indexed table: {table} cols: {len(cols)} pk: {pk_str} fk_count: {len(fks)}")

        except SQLAlchemyError as e:
            print(f"Skipping table {table} due to SQLAlchemy error: {e}")
        except Exception as e:
            print(f"Unexpected error for table {table}: {e}")

    # write to file
    out_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"schema_index_{db_name}.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema_index, f, indent=2, ensure_ascii=False)
    print(f"Wrote schema index to: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Database name (as used in get_engine)")
    args = parser.parse_args()
    build_index_for_db(args.db)
