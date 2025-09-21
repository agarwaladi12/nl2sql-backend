# backend/app/langchain_nl2sql.py
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Configure model
def get_gemini_llm(model="gemini-1.5-flash"):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

SQL_PROMPT = PromptTemplate(
    input_variables=["user_query", "last_sql", "schema_text"],
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

User request: {user_query}

Return ONLY the final SQL query using single quotes for literals.
"""
)

# Track only the *last SQL* for refinement
memory = {"last_sql": ""}

def create_sql_chain(schema_text: str):
    llm = get_gemini_llm()
    output_parser = StrOutputParser()
    chain = SQL_PROMPT | llm | output_parser
    return chain

def run_sql_chain(chain, schema_text, user_query):
    # Inject last_sql into the prompt
    generated_sql = chain.invoke({
        "user_query": user_query,
        "last_sql": memory.get("last_sql", ""),
        "schema_text": schema_text
    }).strip()

    # Update memory with the newly generated SQL
    memory["last_sql"] = generated_sql
    return generated_sql
