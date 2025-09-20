# backend/app/langchain_nl2sql.py
import os
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
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
    input_variables=["user_query", "recent_history", "schema_text"],
    template="""
You are converting a natural language request into a single valid PostgreSQL SELECT statement.
Use only SELECT queries, make string comparisons case-insensitive with LOWER(TRIM(column)).
Prefer table-qualified column names when needed.
In follow-up questions, refine or modify the last SQL query instead of starting over.
Use the sample values provided to match strings exactly as they appear in the database.
Prefer simple, efficient SQL. Use LIMIT when the user asks for a number of results.
Use joins if possible after understanding the schema provided do not include columns in where clause which are not present in that table. Read schema correctly.
Database schema:
{schema_text}

Recent user queries & results:
{recent_history}

User request: {user_query}

Return ONLY the final SQL query using single quotes for literals.
"""
)

# Memory (still used, but you can also fetch from DB instead)
memory = ConversationBufferMemory(
    memory_key="recent_history",
    input_key="user_query",
    return_messages=True
)

def create_sql_chain(schema_text: str):
    llm = get_gemini_llm()
    output_parser = StrOutputParser()

    # RunnableSequence instead of deprecated LLMChain
    chain = SQL_PROMPT | llm | output_parser
    return chain
