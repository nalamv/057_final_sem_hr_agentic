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

# ----- Interactive loop -----
def run_repl(user_input):
    print("Agent ready. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting. Bye!")
                break

            # invoke the agent with the user's message
            try:
                result = agent.invoke({"messages": [HumanMessage(user_input)]})
            except Exception as e:
                # handle runtime/LLM/tool errors gracefully
                print(f"[Error invoking agent] {e}")
                continue

            # extract and print the assistant reply (same approach you used)
            messages = result.get("messages", [])
            if not messages:
                print("[No response returned by the agent]")
                continue

            final_message = messages[-1]
            ai_response_content = getattr(final_message, "content", str(final_message))
            return ai_response_content

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
