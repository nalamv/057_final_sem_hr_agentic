#pip install -U langchain langchain-community langchain-groq langchain-classic sqlalchemy pymysql
import os
from urllib.parse import quote_plus

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

from src.constants import PAYROLL_PROMPT


def create_db_mysql():
    MYSQL_USER = "root"
    MYSQL_PASS = "admin"
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_DB = "dev"

    params = quote_plus("charset=utf8mb4")
    DB_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?{params}"

    db = SQLDatabase.from_uri(DB_URI, include_tables=["employee_payroll_data"],sample_rows_in_table_info=2 )
    return db


# ---------- LLM ----------
def create_llm():
    print("--- Triggered Payroll Agent  ---")
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Set GROQ_API_KEY environment variable")

    llm = ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0
    )
    return llm

def create_agent():
    print(PAYROLL_PROMPT)
    agent_executor = create_sql_agent(
        llm=create_llm(),
        db=create_db_mysql(),
        verbose=True,
        agent_type="openai-tools",
        max_iterations=5,
        system_prompt=PAYROLL_PROMPT
    )
    return agent_executor

def pay_roll_answers(query):
    payroll_agent=create_agent()
    result= payroll_agent.invoke({"input": query})
    return result


