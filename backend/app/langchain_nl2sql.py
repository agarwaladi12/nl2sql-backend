# backend/app/langchain_nl2sql.py
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# -------------------------------
# LLM setup
# -------------------------------
def get_gemini_llm(model="gemini-2.5-flash-lite"):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

# -------------------------------
# SQL Prompt
# -------------------------------
SQL_PROMPT = PromptTemplate(
    input_variables=["user_query", "last_sql", "schema_text", "context"],
    template="""
You are converting a natural language request into a single valid PostgreSQL SELECT statement.

Rules:
- Only generate SELECT queries.
- Use joins whenever queries involve multiple tables, rather than placing unrelated columns in WHERE.
- Always check the schema carefully before using a column.
- Make string comparisons case-insensitive with LOWER(TRIM(column)).
- Prefer table-qualified column names when needed.
- In follow-up questions, refine or modify the LAST SQL query instead of starting over.
- Use LIMIT when the user asks for a number of results.
- Return efficient, readable SQL.

Database schema:
{schema_text}

Last SQL query (if any):
{last_sql}

Relevant past queries/SQL (if useful):
{context}

User request: {user_query}

Return ONLY the final SQL query using single quotes for literals.
"""
)

# -------------------------------
# Memory: multi-turn short-term + long-term
# -------------------------------
memory = {
    "history": [],           # last n queries + SQL
    "max_history": 10        # configurable
}

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
vector_store_path = "faiss_sql_index"
if os.path.exists(vector_store_path):
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = None

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
def run_sql_chain(chain, schema_text, user_query):
    global vector_store  # <--- add this

    # Retrieve last N queries as context
    context_items = [
        f"Q: {item['query']}\nSQL: {item['sql']}"
        for item in memory["history"]
    ]
    context_text = "\n\n".join(context_items)

    # Retrieve long-term memory
    if vector_store is not None:
        docs_and_scores = vector_store.similarity_search_with_score(user_query, k=2)
    else:
        docs_and_scores = []

    if docs_and_scores:
        doc_context, score = docs_and_scores[0]
        context_text += f"\n\n{doc_context.page_content}"
    else:
        score = 0.0

    # Low confidence → ask clarifying question
    first_query = len(memory["history"]) == 0

    # Only ask for clarification if NOT first query
    if not first_query and score < 0.35:
        clarifier = get_gemini_llm().invoke(
            f"Ask a clarifying question to better understand this request: {user_query}"
        )
        return {"message": clarifier.content, "clarification_required": True}
    
    # Otherwise, generate SQL immediately
    generated_sql = chain.invoke({
        "user_query": user_query,
        "last_sql": memory["history"][-1]["sql"] if memory["history"] else "",
        "schema_text": schema_text,
        "context": context_text
    }).strip()

    # Self-reflection: validate before saving
    if validate_sql_with_schema(generated_sql, schema_text):
        # Update short-term memory
        memory["history"].append({"query": user_query, "sql": generated_sql})
        if len(memory["history"]) > memory["max_history"]:
            memory["history"] = memory["history"][-memory["max_history"]:]

        # Update long-term memory
        if vector_store is None:
            vector_store = FAISS.from_texts([f"Q: {user_query}\nSQL: {generated_sql}"], embedding_model)
        else:
            vector_store.add_texts([f"Q: {user_query}\nSQL: {generated_sql}"])
        vector_store.save_local("faiss_sql_index")

        return {"sql": generated_sql, "clarification_required": False}
    else:
        # Ask for clarification instead of guessing
        clarifier = get_gemini_llm().invoke(
            f"Ask a clarifying question to better understand this request: {user_query}"
        )
        return {"message": clarifier.content, "clarification_required": True}
