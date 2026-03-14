import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from src import PolicyAgent, PayrollAgent
from src.constants import LLM_MODEL, API_KEY


@tool
def hr_policies(query: str) -> str:
    """search for the testdata related information like Casual Leave/Sick Leave, Privilege Leave, Maternity/Adoption Leave
    Paternity leave, Loss of Pay Leave, Bereavement Leave, Relocation Transfer Leave"""
    return PolicyAgent.policy_agent001(query)

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location. if any wather related """
    return f"Weather in {location}: Sunny, 72°F"

@tool
def get_payroll_details(query: str)-> str:
    """ Use this tool for any employee related payroll questions like net amount, number of leaves taken, father's name
    Gross salary, deductions like mobile deductions, HRA and insurance cuttings"""
    return PayrollAgent.pay_roll_answers(query)

@tool
def get_payroll_queries(query: str)-> str:
    """Use this agent for any type of Employee Payroll related information, Salaries related queries and
    any employee related tax or number of present days or employee personal information as well"""
    return PolicyAgent.policy_agent001(query)

llm = ChatGroq(groq_api_key=API_KEY, model_name=LLM_MODEL)
agent = create_agent(llm, tools=[hr_policies, get_payroll_details], system_prompt="You are a helpful assistant. Be concise and accurate.")


#result=agent.invoke({"messages":[HumanMessage("How many days for Privilege Leave")]})
#result=agent.invoke({"messages":[HumanMessage("What is the weather of Hyderabad")]})
#result=agent.invoke({"messages":[HumanMessage("what is total employee count in payroll data")]})

result=agent.invoke({"messages":[HumanMessage("what is net pay for employee 'Neha Joshi'")]})
messages = result['messages']
final_message = messages[-1]
ai_response_content = final_message.content

print(ai_response_content)



