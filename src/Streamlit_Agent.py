import asyncio

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from hr_mcp_sever.mcp_client import MCPGroqClient
from src import PolicyAgent, PayrollAgent, HolidayAgent
from src.constants import API_KEY, LLM_MODEL
from hr_mcp_sever import mcp_client


# ----- tools -----
@tool
def hr_policies(query: str) -> str:
    """search for policy info like Casual/Sick/Privilege/Maternity/Paternity leaves etc."""
    return PolicyAgent.policy_agent001(query)

@tool
def hr_holidays(query: str) -> str:
    """search for holidays and give the details of asked holidays"""
    return HolidayAgent.holiday_agent001(query)

@tool
def get_payroll_details(query: str) -> str:
    """employee payroll queries: net amount, leaves, deductions, personal details etc. It can also assist for any questions related to employee records
    You can pick payroll agent if user asked about any father details or department, designation, pay and present days"""
    return PayrollAgent.pay_roll_answers(query)

@tool
def lms_agent_service(query: str) -> str:
    """employee leave management queries: apply for leave, check leave balance, cancel leave, etc
    if user asks about thel list of the employees from Leave management system, LMS also should call this agent."""
    print("---Calling MCPGroqClient for LMS query---")

    async def run_chat():
        client = MCPGroqClient()
        await client.connect("hr_mcp_sever/mcp_server.py")
        response = await client.chat(query)
        await client.close()  # Add this if MCPGroqClient supports cleanup
        return response

    return asyncio.run(run_chat())


# ----- LLM -----
if not API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in environment")

llm = ChatGroq(groq_api_key=API_KEY, model_name=LLM_MODEL)
agent = create_agent(
    llm,
    tools=[hr_policies, get_payroll_details, hr_holidays, lms_agent_service],
    #debug=True,
    system_prompt="You are a helpful assistant. you should provide answers using given tools only. Be concise and accurate."
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
                result = agent.invoke({"messages": chat_history})
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
