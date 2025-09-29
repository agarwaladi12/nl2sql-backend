import re
from typing import Dict, Any, List
from datetime import datetime
import sqlparse


def normalize_schema(raw_schema):
    """
    Normalize schema info into:
      { table_lower: { col_lower: {"nullable": bool, "default": val, "type": str_or_None, "required": bool} } }

    Accepts multiple raw_schema shapes:
    - inspector simple: {table: ["col1", "col2", ...]}
    - inspector detailed: {table: [ {"name": "...", "nullable": ..., "default": ..., "type": ...}, ... ]}
    - schema_index JSON: {"tables": { table: { "columns": [ {...}, ... ] }, ... } }

    This function is defensive and will skip malformed entries.
    """
    schema_map = {}

    if not raw_schema:
        return schema_map

    # If it's a schema_index-like dict with top-level "tables", use that
    if isinstance(raw_schema, dict) and "tables" in raw_schema and isinstance(raw_schema["tables"], dict):
        tables_iter = raw_schema["tables"].items()
    elif isinstance(raw_schema, dict):
        tables_iter = raw_schema.items()
    else:
        raise TypeError("normalize_schema expects a dict-like raw_schema")

    for table_name, cols in tables_iter:
        if table_name is None:
            continue
        tkey = str(table_name).lower()
        schema_map.setdefault(tkey, {})

        # If `cols` is a dict and contains a `columns` key (schema_index style)
        if isinstance(cols, dict) and "columns" in cols and isinstance(cols["columns"], list):
            cols_list = cols["columns"]
        else:
            cols_list = cols

        # If cols_list is not iterable/list, skip
        if not isinstance(cols_list, list):
            continue

        for col in cols_list:
            # Case A: column is a dict with metadata
            if isinstance(col, dict):
                # find name key (support a few possible spellings)
                name = col.get("name") or col.get("column") or col.get("column_name")
                if not name:
                    # malformed column dict, skip
                    continue
                nullable = col.get("nullable", True)
                default = col.get("default", None)
                col_type = col.get("type", None)
            else:
                # Case B: column is a plain string like inspector simple format
                name = str(col)
                nullable = True
                default = None
                col_type = None

            ckey = name.lower()
            required = (not bool(nullable)) and (default is None)

            schema_map[tkey][ckey] = {
                "nullable": bool(nullable),
                "default": default,
                "type": col_type,
                "required": required
            }

    return schema_map

def extract_table_name(sql: str, stmt_type: str) -> str:
    sql = sql.strip()
    if stmt_type == "INSERT":
        # matches: INSERT INTO table_name (...)
        m = re.match(r"INSERT\s+INTO\s+([^\s(]+)", sql, re.IGNORECASE)
        return m.group(1).lower() if m else None
    elif stmt_type == "UPDATE":
        m = re.match(r"UPDATE\s+([^\s(]+)", sql, re.IGNORECASE)
        return m.group(1).lower() if m else None
    elif stmt_type == "DELETE":
        m = re.match(r"DELETE\s+FROM\s+([^\s(]+)", sql, re.IGNORECASE)
        return m.group(1).lower() if m else None
    return None

def cast_value_for_sql(value: str, col_type: str) -> str:
    """Cast a string value to a properly formatted SQL literal based on column type."""
    if value.upper() == "NULL":
        return "NULL"

    col_type = col_type.upper()

    try:
        if "INT" in col_type:
            return str(int(value))
        elif "NUMERIC" in col_type or "DECIMAL" in col_type or "FLOAT" in col_type or "DOUBLE" in col_type:
            return str(float(value))
        elif "DATE" in col_type:
            # Convert to ISO date format string
            dt = datetime.fromisoformat(value.strip("'").strip('"'))
            return f"'{dt.date().isoformat()}'"
        elif "CHAR" in col_type or "TEXT" in col_type or "VARCHAR" in col_type:
            val = value.strip("'").strip('"')
            return f"'{val}'"
        else:
            return value
    except Exception:
        # fallback: quote as string
        val = value.strip("'").strip('"')
        return f"'{val}'"

def validate_and_cast_dml(sql: str, schema_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates a DML SQL statement against a normalized schema,
    casts values to proper SQL literals, and returns updated SQL.
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return {"valid": False, "message": "Unable to parse SQL.", "sql": sql}

    stmt = parsed[0]
    stmt_type = stmt.get_type()
    if stmt_type not in ("INSERT", "UPDATE", "DELETE"):
        return {"valid": False, "message": "Not a DML query.", "sql": sql}

    # Extract table name
    table_name = extract_table_name(sql, stmt_type)
    if not table_name or table_name not in schema_map:
        return {"valid": False, "message": f"Table '{table_name}' not found in schema.", "sql": sql}

    table_cols = schema_map[table_name]
    missing_required = []

    if stmt_type == "INSERT":
        # Extract columns and values
        m = re.match(r"INSERT\s+INTO\s+\w+\s*\((.*?)\)\s*VALUES\s*\((.*?)\)", sql, re.IGNORECASE)
        if not m:
            return {"valid": False, "message": "INSERT statement parsing failed.", "sql": sql}

        cols = [c.strip().lower() for c in m.group(1).split(",")]
        values = [v.strip() for v in m.group(2).split(",")]

        # check missing required
        for col, meta in table_cols.items():
            if not meta.get("nullable", True) and col not in cols and meta.get("default") is None:
                missing_required.append(col)

        # cast values
        casted_values = []
        for col, val in zip(cols, values):
            if col in table_cols:
                casted_values.append(cast_value_for_sql(val, table_cols[col]["type"]))
            else:
                casted_values.append(val)

        # rebuild SQL
        new_sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(casted_values)});"

    elif stmt_type == "UPDATE":
        # Extract SET clauses
        set_match = re.search(r"SET\s+(.*?)(\s+WHERE|$)", sql, re.IGNORECASE | re.DOTALL)
        set_text = set_match.group(1) if set_match else ""
        assignments = [a.strip() for a in set_text.split(",") if a.strip()]
        new_assignments = []

        for a in assignments:
            if "=" in a:
                col, val = a.split("=", 1)
                col = col.strip().lower()
                val = val.strip()
                if col in table_cols:
                    val = cast_value_for_sql(val, table_cols[col]["type"])
                new_assignments.append(f"{col} = {val}")
            else:
                new_assignments.append(a)

        where_clause = sql[sql.upper().find("WHERE"):] if "WHERE" in sql.upper() else ""
        new_sql = f"UPDATE {table_name} SET {', '.join(new_assignments)} {where_clause}"

    else:  # DELETE
        new_sql = sql

    if missing_required:
        return {
            "valid": False,
            "message": f"Missing required columns: {', '.join(missing_required)}",
            "sql": sql
        }

    return {"valid": True, "message": "SQL is valid and values casted.", "sql": new_sql}