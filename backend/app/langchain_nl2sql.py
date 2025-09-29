# backend/app/langchain_nl2sql.py
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# -------------------------------
# LLM setup
# -------------------------------
def get_gemini_llm(model="gemini-2.5-flash"):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

# -------------------------------
# SQL Prompt (supports SELECT & DML)
# -------------------------------
SQL_PROMPT = PromptTemplate(
    input_variables=["user_query", "last_sql", "schema_text", "context"],
    template="""
You are converting a natural language request into a valid PostgreSQL SQL query.

Rules:
- Generate SELECT, INSERT, UPDATE, DELETE, MERGE/UPSERT queries as required.
- Use joins whenever queries involve multiple tables.
- Always check the schema before using columns.
- Make string comparisons case-insensitive with LOWER(TRIM(column)).
- Table-qualified column names preferred when needed.
- Use LIMIT when user requests number of results.
- Return efficient, readable SQL.
Hard Rules:
1. Always use the provided database schema to validate table and column names.
2. For INSERT or UPDATE:
   - Always include all columns marked as required in the schema. 
   - For required columns, generate realistic placeholder data if not provided in the user query.
   - Never omit required columns, even if the user only specifies one field.
3. Do not ask clarifying questions. If user input is incomplete, still generate a valid SQL using placeholders.
4. Provide **structured JSON output** with two keys:
   - "sql": the final SQL query.
   - "suggestions": a list of suggestions to improve the SQL (e.g., include indexes, optimize joins, avoid NULLs if possible).

Example JSON output:
{{
    "sql": "INSERT INTO country (code, name, continent) VALUES ('TST', 'TestLand', 'Asia');",
    "suggestions": [
        "Consider adding population and capital columns if required by schema.",
        "Ensure the country code is unique to avoid conflicts."
    ]
}}

Database schema:
{schema_text}

Last SQL query (if any):
{last_sql}

Relevant past queries/SQL (if useful):
{context}

User request: {user_query}

Return **ONLY valid JSON** as shown in the example.
"""
)

# -------------------------------
# Memory: multi-turn short-term + long-term
# -------------------------------
memory = {"history": [], "max_history": 10}

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
vector_store_path = "faiss_sql_index"
if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = None

memory = {}
DEFAULT_MAX_HISTORY = 10

# -------------------------------
# Chain creator
# -------------------------------
def create_sql_chain(schema_text: str):
    llm = get_gemini_llm()
    output_parser = StrOutputParser()
    chain = SQL_PROMPT | llm | output_parser
    return chain

# -------------------------------
# SQL Validation
# -------------------------------
def validate_sql_with_schema(sql: str, schema_text: str) -> bool:
    validator = get_gemini_llm()
    check_prompt = f"""
    Here is a PostgreSQL query:
    {sql}

    And here is the schema:
    {schema_text}

    Question: Does this SQL correctly use only columns/tables from the schema,
    follow correct SQL syntax, and align with the request? Answer only 'YES' or 'NO'.
    """
    response = validator.invoke(check_prompt)
    return "YES" in response.content.strip().upper()

# -------------------------------
# Run SQL chain
# -------------------------------
def run_sql_chain(chain, schema_text, user_query, user_id):
    global memory, vector_store

    # Ensure per-user session history
    if user_id not in memory:
        memory[user_id] = {"history": [], "max_history": DEFAULT_MAX_HISTORY}

    session_history = memory[user_id]["history"]

    # Short-term context (with flags)
    context_items = [
        f"Q: {item['query']}\nSQL: {item['sql']}\nExecuted: {item.get('executed', False)}\nSuggestions: {item.get('suggestions', [])}"
        for item in session_history
    ]
    context_text = "\n\n".join(context_items)

    # Long-term context
    docs_and_scores = vector_store.similarity_search_with_score(user_query, k=2) if vector_store else []
    if docs_and_scores:
        doc_context, score = docs_and_scores[0]
        context_text += f"\n\n{doc_context.page_content}"

    # Prompt for LLM
    gemini_prompt = {
        "user_query": user_query,
        "last_sql": session_history[-1]["sql"] if session_history else "",
        "schema_text": schema_text,
        "context": context_text
    }
    generated_text = chain.invoke(gemini_prompt).strip()

    # Parse JSON output
    try:
        import json
        result = json.loads(generated_text)
        sql = result.get("sql", "")
        suggestions = result.get("suggestions", [])
    except Exception:
        sql = generated_text
        suggestions = []

    # Detect DML (INSERT/UPDATE/DELETE)
    is_dml = sql.strip().lower().startswith(("insert", "update", "delete"))

    # Validate SQL
    if validate_sql_with_schema(sql, schema_text):
        # Update vector store
        if vector_store is None:
            vector_store = FAISS.from_texts([f"Q: {user_query}\nSQL: {sql}"], embedding_model)
        else:
            vector_store.add_texts([f"Q: {user_query}\nSQL: {sql}"])
        vector_store.save_local("faiss_sql_index")

        return {
            "sql": sql,
            "suggestions": suggestions,
            "clarification_required": False,
            "requires_confirmation": is_dml
        }
    else:
        return {
            "sql": sql,
            "suggestions": suggestions,
            "clarification_required": True,
            "requires_confirmation": is_dml
        }