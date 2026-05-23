import os
import json
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
#from langchain_core.pydantic_v1 import Field

# --- 1. CONFIGURATION ---
MYSQL_USER = "root"
MYSQL_PASS = "admin"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "dev"
TABLE_NAME = "employee_payroll_data"
SCHEMA_FILE = "db_schema.json"

params = quote_plus("charset=utf8mb4")
DB_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?{params}"


# --- 2. CACHING AND UTILITY FUNCTIONS ---

def get_db_schema(uri, table_names):
    """Connects to DB, introspects schema, and returns the required information."""
    print("Connecting to DB and performing schema introspection (This may take a moment)...")
    db = SQLDatabase.from_uri(uri, include_tables=table_names, sample_rows_in_table_info=0)

    # Get the table info string used as context for the LLM
    table_info = db.get_table_info(table_names)

    return table_info


def load_or_generate_schema(uri, table_name, schema_file):
    """Checks for cached schema; generates and saves it if not found."""
    if os.path.exists(schema_file):
        print(f"✅ Loading cached schema from {schema_file}...")
        with open(schema_file, "r") as f:
            data = json.load(f)
            return data["table_info"]
    else:
        # Schema file not found, perform slow introspection
        schema_data = get_db_schema(uri, [table_name])

        # Save the result for next time
        with open(schema_file, "w") as f:
            json.dump({"table_info": schema_data}, f)

        print(f"🎉 Schema generated and saved to {schema_file}.")
        return schema_data


# --- 3. CUSTOM TOOLKIT (The Fix) ---

class CachedSQLDatabaseToolkit(SQLDatabaseToolkit):
    """
    A custom SQLDatabaseToolkit that uses a pre-cached schema string
    instead of running slow DB introspection when creating the tool's description.
    """
    cached_table_info: str = Field(...)
    def __init__(self, db: SQLDatabase, llm: BaseTool, cached_table_info: str):
        super().__init__(db=db, llm=llm)
        self.cached_table_info = cached_table_info

    def get_tools(self) -> list[BaseTool]:
        """Override to provide custom schema information."""

        # Get the standard tools (Query, Schema, List Tables)
        tools = super().get_tools()

        # Find the tool that provides the schema (usually the 'sql_db_schema' tool)
        for tool in tools:
            if tool.name == "sql_db_schema":
                # Crucial Fix: Override the description of the schema tool
                # to include the cached schema string directly.
                tool.description = (
                    "Input is a comma-separated list of tables names, output is the "
                    "SQL CREATE TABLE and sample rows statements for those tables.\n"
                    f"--- CACHED SCHEMA INFO ---\n{self.cached_table_info}"
                )
        return tools


# --- 4. MAIN EXECUTION ---

# Load or generate the cached schema text
cached_table_info = load_or_generate_schema(DB_URI, TABLE_NAME, SCHEMA_FILE)

# Create the SQLAlchemy Engine (required for the agent to execute queries)
engine = create_engine(DB_URI)

# Create the SQLDatabase object (fast, only sets up connection)
db = SQLDatabase(engine=engine, include_tables=[TABLE_NAME])

# Initialize the LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Set GROQ_API_KEY environment variable")

llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)
# Create the custom toolkit using the cached schema
toolkit = CachedSQLDatabaseToolkit(
    db=db,
    llm=llm,
    cached_table_info=cached_table_info
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


# Create the SQL Agent using the custom toolkit
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,  # Pass the toolkit directly
    agent_type="openai-tools",
    verbose=True,
    max_iterations=5,
    system_prompt=AVOID_DELETE_ACTIONS
)

# --- 5. USAGE EXAMPLE ---
print("\n--- Agent Setup Complete ---")
print(f"Usable Table Names: {db.get_usable_table_names()}")
print("\nAgent is ready to answer queries.")

response = agent_executor.invoke({"input": "What is the average Net_Amount in the payroll data?"})
print(response)