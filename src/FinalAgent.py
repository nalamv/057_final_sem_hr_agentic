import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from torch.backends.mkl import verbose

from src import PolicyAgent, PayrollAgent, HolidayAgent
from src.constants import API_KEY, LLM_MODEL


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


# ----- LLM -----
if not API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in environment")

llm = ChatGroq(groq_api_key=API_KEY, model_name=LLM_MODEL)
agent = create_agent(
    llm,
    tools=[hr_policies, get_payroll_details, hr_holidays],
    #debug=True,
    system_prompt="You are a helpful assistant. you should provide answers using given tools only. Be concise and accurate."
)

# ----- Interactive loop -----
def run_repl():
    print("Agent ready. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        while True:
            user_input = input("You: ").strip()
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
            print("\nAssistant:", ai_response_content)
            print("-" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")

if __name__ == "__main__":
    run_repl()
