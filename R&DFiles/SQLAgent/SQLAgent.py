import os
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
#from langchain.agents import AgentType

# --- 1. LLM and API Configuration for Grok AI ---

# IMPORTANT: Replace these with your actual Grok AI service details
GROK_API_BASE = "YOUR_GROK_AI_SERVICE_URL/v1"  # e.g., http://localhost:8000/v1
GROK_MODEL_NAME = "grok-ai/gpt-oss-20b"
# If your Grok AI service requires an API key, set it here or as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_GROK_API_KEY"

# Initialize the LLM client, configured for your Grok AI endpoint
try:
    groq_api_key = "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"
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
    # llm = ChatOpenAI(
    #     model=GROK_MODEL_NAME,
    #     openai_api_base=GROK_API_BASE,
    #     temperature=0,  # Lower temperature for deterministic SQL generation
    #     # Add your API key if required by your Grok service
    #     # api_key=os.environ.get("OPENAI_API_KEY", "dummy-key")
    # )
except Exception as e:
    print(f"Error initializing LLM. Check API base URL and key. Error: {e}")
    llm = None  # Set to None if initialization fails

# --- 2. SQL Database Setup ---

# Replace with your actual database connection string
# Example MySQL: "mysql+mysqlconnector://user:pass@host/dbname"
# Example MSSQL: "mssql+pyodbc://user:pass@dsn_name"
# Example SQLite (simple): "sqlite:///./my_database.db"
DATABASE_URI = "mysql+mysqlconnector://root:admin@localhost:3306/dev"

try:
    # Use the SQLDatabase wrapper to allow LangChain to introspect the schema
    db = SQLDatabase.from_uri(DATABASE_URI)
    print(f"Successfully connected to database: {DATABASE_URI}")
except Exception as e:
    print(f"Error connecting to database. Ensure URI is correct. Error: {e}")
    db = None


# --- 3. SQL Agent Design Utility ---

def query_sql_database(question: str):
    """
    Designs a utility to answer natural language questions
    by generating and executing SQL queries via an LLM agent.
    """
    if not llm or not db:
        return "System configuration failed (LLM or Database connection)."

    print("\n--- Initializing SQL Agent ---")

    # Create the SQL Agent
    # We use AgentType.OPENAI_FUNCTIONS for best performance and structured output,
    # assuming your Grok AI service supports function calling (OpenAI format).
    # If not, use AgentType.ZERO_SHOT_REACT_DESCRIPTION (less reliable).
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",  # or AgentType.ZERO_SHOT_REACT_DESCRIPTION
        verbose=True,  # Set to True to see the thought process and SQL query
        top_k=5  # Limit the number of query results to return
    )

    print(f"--- Processing Query: {question} ---")

    # Run the query through the agent
    try:
        response = agent_executor.invoke({"input": question})
        return response.get('output', "Could not retrieve an answer.")
    except Exception as e:
        return f"An error occurred during agent execution: {e}"


# --- 4. Example Usage ---

if __name__ == "__main__":
    # 🚨 NOTE: Before running, ensure you have a simple table (e.g., 'Employee_Payroll_Data')
    # in your SQLite database for the agent to query.

    # Example 1: Simple Aggregation
    query1 = "What is the total basic salary across all employees?"
    answer1 = query_sql_database(query1)
    print("\n[Query 1 Result]:")
    print(answer1)

    # Example 2: Retrieval based on condition
    query2 = "Which employee has the designation 'Software Engineer' and what is their Father's Name?"
    answer2 = query_sql_database(query2)
    print("\n[Query 2 Result]:")
    print(answer2)