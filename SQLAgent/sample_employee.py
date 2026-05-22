# =========================================================
# INSTALL REQUIRED PACKAGES
# =========================================================
# pip install -U \
# langchain \
# langchain-core \
# langchain-community \
# langchain-groq \
# sqlalchemy \
# pymysql


# =========================================================
# IMPORTS
# =========================================================

import os
import re

from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

MYSQL_USER = "root"
MYSQL_PASS = "admin"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "dev"


# =========================================================
# CREATE DATABASE ENGINE
# =========================================================

def create_db_engine():

    params = quote_plus("charset=utf8mb4")

    DB_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?{params}"
    )

    engine = create_engine(DB_URI)

    return engine


# =========================================================
# GET TABLE SCHEMA
# =========================================================

def get_table_schema(engine):

    inspector = inspect(engine)

    columns = inspector.get_columns(
        "employee_payroll_data"
    )

    schema = """
Table Name: employee_payroll_data

Columns:
"""

    for col in columns:
        schema += f"- {col['name']}\n"

    return schema


# =========================================================
# CREATE LLM
# =========================================================

def create_llm():

    print("\n--- Triggered Payroll AI Assistant ---\n")

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not found"
        )

    llm = ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return llm


# =========================================================
# CLEAN SQL RESPONSE
# =========================================================

def clean_sql_response(sql_query):

    sql_query = sql_query.strip()

    # Remove markdown formatting
    sql_query = sql_query.replace("```sql", "")
    sql_query = sql_query.replace("```", "")

    sql_query = sql_query.strip()

    return sql_query


# =========================================================
# GENERATE SECURE SQL
# =========================================================

def generate_secure_sql(
    user_query,
    employee_id,
    schema
):

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a secure payroll SQL generator.

Database Schema:
{schema}

IMPORTANT SECURITY RULES:

1. ALWAYS query ONLY from:
   employee_payroll_data

2. NEVER use employee names in SQL.

3. ALWAYS use:
   WHERE Emp_ID = '{employee_id}'

4. ONLY use valid columns from schema.

5. include the employee name and employee id 

6. NEVER generate:
   DELETE
   DROP
   UPDATE
   INSERT
   ALTER
   TRUNCATE

7. ONLY generate SELECT queries.

8. Ignore employee names mentioned in user query.

9. Return ONLY raw SQL query.
   No explanation.
   No markdown.

User Question:
{user_query}
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_query": user_query,
        "employee_id": employee_id,
        "schema": schema
    })

    sql_query = response.content

    sql_query = clean_sql_response(sql_query)

    print("\n==============================")
    print("GENERATED SQL")
    print("==============================")
    print(sql_query)

    return sql_query


# =========================================================
# VALIDATE SQL
# =========================================================

def validate_sql(
    sql_query,
    employee_id,
    schema
):

    sql_lower = sql_query.lower()

    # ------------------------------------------------
    # ONLY SELECT
    # ------------------------------------------------
    if not sql_lower.startswith("select"):
        raise Exception(
            "Only SELECT queries are allowed"
        )

    # ------------------------------------------------
    # BLOCK DANGEROUS SQL
    # ------------------------------------------------
    blocked_keywords = [
        "drop",
        "delete",
        "truncate",
        "update",
        "insert",
        "alter"
    ]

    for keyword in blocked_keywords:

        if keyword in sql_lower:
            raise Exception(
                f"Blocked dangerous keyword: {keyword}"
            )

    # ------------------------------------------------
    # ENSURE EMPLOYEE FILTER
    # ------------------------------------------------
    required_filter = (
        f"emp_id = '{employee_id.lower()}'"
    )

    if required_filter not in sql_lower:
        raise Exception(
            "Unauthorized query detected"
        )

    # ------------------------------------------------
    # VALIDATE TABLE NAME
    # ------------------------------------------------
    if "employee_payroll_data" not in sql_lower:
        raise Exception(
            "Unauthorized table access"
        )

    print("\n==============================")
    print("SQL VALIDATION SUCCESS")
    print("==============================")

    return True


# =========================================================
# EXECUTE SQL
# =========================================================

def execute_sql(engine, sql_query):

    print("\n==============================")
    print("EXECUTING SQL")
    print("==============================")
    print(sql_query)

    try:

        with engine.connect() as conn:

            result = conn.execute(text(sql_query))

            rows = result.fetchall()

            return rows

    except Exception as e:

        print("\n==============================")
        print("SQL EXECUTION ERROR")
        print("==============================")
        print(str(e))

        raise e


# =========================================================
# GENERATE FINAL ANSWER
# =========================================================

def generate_final_answer(
    user_query,
    db_result
):

    llm = create_llm()

    # Handle empty results safely
    if not db_result:

        return (
            "No matching payroll information "
            "was found for the authenticated employee."
        )

    prompt = ChatPromptTemplate.from_template("""
You are a payroll assistant.

Answer the question ONLY from the database result.

User Question:
{user_query}

Database Result:
{db_result}

Provide a short and clear response.
""")

    chain = prompt | llm

    response = chain.invoke({
        "user_query": user_query,
        "db_result": str(db_result)
    })

    return response.content


# =========================================================
# MAIN PAYROLL AGENT FLOW
# =========================================================

def payroll_agent(
    user_query,
    employee_id
):

    print("\n==============================")
    print("USER QUERY")
    print("==============================")
    print(user_query)

    # ------------------------------------------------
    # CREATE ENGINE
    # ------------------------------------------------
    engine = create_db_engine()

    # ------------------------------------------------
    # FETCH SCHEMA
    # ------------------------------------------------
    schema = get_table_schema(engine)

    print("\n==============================")
    print("DATABASE SCHEMA")
    print("==============================")
    print(schema)

    # ------------------------------------------------
    # GENERATE SQL
    # ------------------------------------------------
    sql_query = generate_secure_sql(
        user_query=user_query,
        employee_id=employee_id,
        schema=schema
    )

    # ------------------------------------------------
    # VALIDATE SQL
    # ------------------------------------------------
    validate_sql(
        sql_query=sql_query,
        employee_id=employee_id,
        schema=schema
    )

    # ------------------------------------------------
    # EXECUTE SQL
    # ------------------------------------------------
    db_result = execute_sql(
        engine=engine,
        sql_query=sql_query
    )

    print("\n==============================")
    print("DATABASE RESULT")
    print("==============================")
    print(db_result)

    # ------------------------------------------------
    # GENERATE FINAL ANSWER
    # ------------------------------------------------
    final_answer = generate_final_answer(
        user_query=user_query,
        db_result=db_result
    )

    return {
        "sql_query": sql_query,
        "db_result": db_result,
        "final_answer": final_answer
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # ------------------------------------------------
    # AUTHENTICATED EMPLOYEE ID
    # ------------------------------------------------
    current_employee_id = "116"

    # ------------------------------------------------
    # USER QUERY
    # ------------------------------------------------
    user_query = (
        "Give me the break down of all salary of Neha?"
    )

    # ------------------------------------------------
    # EXECUTE PAYROLL AGENT
    # ------------------------------------------------
    response = payroll_agent(
        user_query=user_query,
        employee_id=current_employee_id
    )

    # ------------------------------------------------
    # FINAL RESPONSE
    # ------------------------------------------------
    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(response["final_answer"])