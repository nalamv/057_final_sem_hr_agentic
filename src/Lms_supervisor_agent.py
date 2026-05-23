import asyncio
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from hr_mcp_sever.mcp_client import MCPGroqClient
from hr_mcp_sever.mcp_employee_client import MCPGroqClient_Employee
from src import PolicyAgent, PayrollAgent, HolidayAgent, TaxInfoAgent
from src.constants import API_KEY, LLM_MODEL

@tool
def lms_manager_agent_service(query: str) -> str:
    """employee leave management queries: apply for leave, check leave balance, cancel leave, etc
    if user asks about the list of the employees from Leave management system, LMS also should call this agent.
    if user asked about any employee;s related questions without providing employee id then, you have call ''get_all_employees' tool and get employee id"""
    print("---Calling manager Specific Tools ---")

    async def run_chat():
        client = MCPGroqClient()
        await client.connect("hr_mcp_sever/mcp_server.py")
        response = await client.chat(query)
        await client.close()  # Add this if MCPGroqClient supports cleanup
        return response

    return asyncio.run(run_chat())

@tool
def lms_employee_agent_service(query: str) -> str:
    """employee leave management queries: apply for leave, check leave balance, etc
   this agent supports for apply_leave and get_employee_leaves functionality only."""
    print("---Calling Employee specific MCP Service---")
    async def run_chat():
        client = MCPGroqClient_Employee()
        try:
            await client.connect("hr_mcp_sever/mcp_server_employee.py")
            return await client.chat(query)
        finally:
            await client.close()  # Add this if MCPGroqClient supports cleanup
    return asyncio.run(run_chat())

def lms_agent(query,emp_type,employee_id=None):
    try:
        llm = ChatGroq(groq_api_key=API_KEY, model_name=LLM_MODEL)
        print(f"Employee Type:{emp_type}, Employee Id: {employee_id}")
        agent_prompt=None
        if emp_type == "employee":
            tools = [lms_employee_agent_service]
            agent_prompt = (
                "You are a helpful assistant for employee leave management system. you should provide answers using given tools only. "
                "Be concise and accurate. You can assist employees for applying leave and checking their leave balance. "
                f"Employee id is {employee_id}")
        else:
            tools = [lms_manager_agent_service]
            agent_prompt = (
                f"You are a helpful assistant for manager in leave management system. you should provide answers using given tools only. "
                "Be concise and accurate. You can assist manager for applying leave, checking leave balance, cancelling leave, approving leave and get employee list from LMS etc. "
                "If user asked about any employee related questions without providing employee id then, you have to call 'get_all_employees' tool and get employee id first before providing the answer. "
                f"Manager Id is {employee_id}")
        supervisor_agent = create_agent(
            llm,
            tools=tools,
            debug=True,
            system_prompt=agent_prompt
        )
        response = supervisor_agent.invoke({"messages": query})
        return response
    except Exception as e:
        print(e)
        return f"Exception Occurred in Policy Agent. {e}"