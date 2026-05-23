"""
Working SQL Agent for MySQL + Groq (LangChain v1.0+ - RECOMMENDED)
"""
#pip install -U langchain langchain-community langchain-groq langchain-classic sqlalchemy pymysql

import os
from urllib.parse import quote_plus

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# ---------- Database ----------
MYSQL_USER = "root"
MYSQL_PASS = "admin"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "dev"

params = quote_plus("charset=utf8mb4")
DB_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?{params}"

db = SQLDatabase.from_uri(DB_URI, include_tables=["employee_payroll_data"],sample_rows_in_table_info=0 )

# ---------- LLM ----------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Set GROQ_API_KEY environment variable")

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)

AVOID_DELETE_ACTIONS=f"""You are a careful SQLite analyst.
Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- Limit to 5 rows of output unless the user explicitly asks otherwise.
- If the tool returns 'Error:', revise the SQL and try again.
- Consider 0 values as Null values in data table 
- Prefer explicit column lists; avoid SELECT *."""


# ---------- Create SQL Agent (handles everything automatically) ----------
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=5,
    system_prompt=AVOID_DELETE_ACTIONS
)

# ---------- Ask a question ----------
#question = "what is the gross salary of Reshma Patil"
question = "how many employess have mobile deduction?"
#question = "what is the salary of Reshma Patil"
#question = "Show the top 5 employees with highest Net Amount"

result = agent_executor.invoke({"input": question})
print("\nFINAL ANSWER:\n", result["output"])


