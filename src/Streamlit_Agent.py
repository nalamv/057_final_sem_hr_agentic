import asyncio

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from hr_mcp_sever.mcp_client import MCPGroqClient
from src import PolicyAgent, PayrollAgent, HolidayAgent, TaxInfoAgent, NewPayrollAgent
from src.constants import API_KEY, LLM_MODEL
from hr_mcp_sever import mcp_client
from src.Lms_supervisor_agent import lms_agent

employee_type=None
employee_id=None
# ----- tools -----
@tool
def hr_policies(query: str) -> str:
    """search for policy info like Casual/Sick/Privilege/Maternity/Paternity leaves etc.
    """
    return PolicyAgent.policy_agent001(query)

@tool
def hr_holidays(query: str) -> str:
    """search for holidays and give the details of asked holidays"""
    return HolidayAgent.holiday_agent001(query)

@tool
def hr_tax_benefits(query: str) -> str:
    """Search for any Tax benefits or Tax related queries of the employees"""
    return TaxInfoAgent.tax_agent_invoke(query)

@tool
def get_payroll_details(query: str) -> str:
    """employee payroll queries: net amount, leaves, deductions, personal details etc. It can also assist for any questions related to employee records
    You can pick payroll agent if user asked about any father details or department, designation, pay and present days"""
    #return PayrollAgent.pay_roll_answers(query)
    return NewPayrollAgent.payroll_agent(query,employee_id)

# @tool
# def lms_manager_agent_service(query: str) -> str:
#     """employee leave management queries: apply for leave, check leave balance, cancel leave, etc
#     if user asks about thel list of the employees from Leave management system, LMS also should call this agent."""
#     print("---Calling MCPGroqClient for LMS query---")
#
#     async def run_chat():
#         client = MCPGroqClient()
#         await client.connect("hr_mcp_sever/mcp_server.py")
#         response = await client.chat(query)
#         await client.close()  # Add this if MCPGroqClient supports cleanup
#         return response
#
#     return asyncio.run(run_chat())
#
# @tool
# def lms_employee_agent_service(query: str) -> str:
#     """employee leave management queries: apply for leave, check leave balance, etc
#    this agent supports for apply_leave and get_employee_leaves functionality only."""
#     print("---Calling Employee specific MCP tools---")
#
#     async def run_chat():
#         client = MCPGroqClient()
#         await client.connect("hr_mcp_sever/mcp_server_employee.py")
#         response = await client.chat(query)
#         print(response)
#         await client.close()  # Add this if MCPGroqClient supports cleanup
#         return response
#
#     return asyncio.run(run_chat())
@tool
def lms_agent_service(query: str) -> str:
    """employee leave management queries: apply for leave, check leave balance, cancel leave, approve leave get employee list from LMS  etc
   """
    return lms_agent(query,employee_type,employee_id)

# ----- LLM -----
if not API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in environment")

llm = ChatGroq(groq_api_key=API_KEY, model_name=LLM_MODEL)
supervisor_agent = create_agent(
    llm,
    tools=[hr_policies, get_payroll_details, hr_holidays, lms_agent_service, hr_tax_benefits],
    #debug=True,
    system_prompt=(f"You are a helpful assistant. you should provide answers using given tools only. Be concise and accurate. Your current role is {employee_type} and employee id is {employee_id}. "
                  f"You have capability to answer the questions related to HR policies, payroll details, holidays, tax benefits and leave management system.")
)
chat_history: list = []

# ----- Interactive loop -----
def run_repl(user_input,chat_history):
    #print("Agent ready. Type your question and press Enter.")
    #print("Type 'exit' or 'quit' to stop.\n")
    if not user_input:
        return "", chat_history

    try:
        while True:
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting. Bye!")
                break

            chat_history.append(HumanMessage(content=user_input))
            chat_history = chat_history[-10:]  # Limit to last 10 messages
            try:
                result = supervisor_agent.invoke({"messages": chat_history})
                messages = result.get("messages", [])
                if messages:
                    chat_history.append(messages[-1])  # Append the AI response message
                    chat_history = chat_history[-10:]  # Ensure limit after append
                else:
                    return "[No response returned by the agent]", chat_history
            except Exception as e:
                return f"[Error invoking agent] {e}", chat_history

            messages = result.get("messages", [])
            if not messages:
                return "[No response returned by the agent]", chat_history

            final_message = messages[-1]
            ai_response_content = getattr(final_message, "content", str(final_message))
            return ai_response_content, chat_history

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")

def write_employee_type(emp_type):
    global employee_type
    employee_type = emp_type
    print(f"DEBUG: Streamlit_Agent.employee_type set to: {employee_type}")

def write_employee_id(emp_id):
    """Set the employee ID at runtime"""
    global employee_id
    employee_id = emp_id
    print(f"DEBUG: Streamlit_Agent.employee_id set to: {employee_id}")