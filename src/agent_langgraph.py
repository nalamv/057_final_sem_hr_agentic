import asyncio
from typing import Annotated
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from hr_mcp_sever.mcp_client import MCPGroqClient
from src import PolicyAgent, PayrollAgent, HolidayAgent, TaxInfoAgent, NewPayrollAgent
from src.constants import API_KEY, LLM_MODEL
from src.Lms_supervisor_agent import lms_agent

employee_type = None
employee_id = None


# ----- Define Agent State -----
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    final_response: str


# ----- Tools -----
@tool
def hr_policies(query: str) -> str:
    """search for policy info like Casual/Sick/Privilege/Maternity/Paternity leaves etc."""
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
    return NewPayrollAgent.payroll_agent(query, employee_id)


@tool
def lms_agent_service(query: str) -> str:
    """employee leave management queries: apply for leave, check leave balance, cancel leave, approve leave get employee list from LMS etc"""
    return lms_agent(query, employee_type,employee_id)

# Create tools list
tools = [hr_policies, hr_holidays, hr_tax_benefits, get_payroll_details, lms_agent_service]

# ----- Initialize LLM -----
if not API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in environment")

SYSTEM_PROMPT = (
    "You are a helpful HR assistant. Use only the provided tools to get facts. "
    "When you call a tool, wait for its results, then reason and call more tools if needed. "
    "Answer concisely and accurately. If you cannot answer, say so and explain what tool/data is missing."
    "You are allowed to use payroll, policy, holiday, tax info and lms tools to answer employee queries. Always use the tools when relevant information is needed to answer employee queries. Do not make up information that can be obtained from the tools. Always use the tools when relevant information is needed to answer employee queries. Do not make up information that can be obtained from the tools."
    "Whenever required you can use Payroll tool for any Employee Tax calculations"
    f"Your Employee Type: {employee_type} and Employee Id: {employee_id}"
)


llm = ChatGroq(groq_api_key=API_KEY, model_name=LLM_MODEL)
llm_with_tools = llm.bind_tools(tools)


# ----- Agent Node -----
def agent_node(state: AgentState) -> AgentState:
    """Call the LLM with tools. Ensure the SYSTEM_PROMPT is present and preserve history."""
    # preserve previous messages (history)
    history = state.get("messages", []) or []
    # keep last 9 user/ai/tool messages plus 1 system prompt if present (total context <= 10)
    history_tail = history[-9:]  # leave room for system prompt
    # ensure system prompt is first message
    if not history_tail or not isinstance(history_tail[0], SystemMessage):
        messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + history_tail
    else:
        messages_for_llm = history_tail

    # call LLM (it will inspect messages_for_llm and decide tool calls)
    response = llm_with_tools.invoke(messages_for_llm)

    # append the LLM response to the history and return full state
    new_history = history + [response]
    # keep history bounded if desired
    new_history = new_history[-50:]  # keep last 50 for state, tune as needed
    return {"messages": new_history, "final_response": state.get("final_response", "")}



# ----- Tool Executor Node -----
def tool_executor_node(state: AgentState) -> AgentState:
    """Execute the tools called by the agent and append ToolMessage results to history."""
    history = state.get("messages", []) or []
    last_message = history[-1] if history else None

    # No tool calls => return state unchanged
    if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state

    tool_calls = last_message.tool_calls
    tool_results_msgs = []

    tool_map = {t.name: t for t in tools}

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        try:
            if tool_name in tool_map:
                tool_result = tool_map[tool_name].invoke(tool_input)
                tool_results_msgs.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call.get("id"),
                        name=tool_name
                    )
                )
            else:
                tool_results_msgs.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not found",
                        tool_call_id=tool_call.get("id"),
                        name=tool_name,
                        is_error=True
                    )
                )
        except Exception as e:
            tool_results_msgs.append(
                ToolMessage(
                    content=f"[tool error] {e}",
                    tool_call_id=tool_call.get("id"),
                    name=tool_name,
                    is_error=True
                )
            )

    # Append tool outputs to history so the agent sees them on next agent_node call
    new_history = history + tool_results_msgs
    new_history = new_history[-50:]
    return {"messages": new_history, "final_response": state.get("final_response", "")}


# ----- Router to determine next action -----
def should_continue(state: AgentState) -> str:
    """Determine if agent should continue or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, continue to tool executor
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end (agent has provided final response)
    return "end"


# ----- Build the Graph -----
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_executor_node)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tool execution, go back to agent for reasoning
workflow.add_edge("tools", "agent")

# Compile the graph
graph = workflow.compile()


# ----- Main interaction function -----
def run_repl(user_input: str, chat_history: list,emp_id:str, emp_type:str) -> tuple[str, list]:
    """Run the agent with LangGraph"""
    if not user_input:
        return "", chat_history

    try:
        # Add user message to history
        chat_history.append(HumanMessage(content=user_input))

        # Keep only last 10 messages
        messages = chat_history[-10:]

        # Initialize state
        initial_state = {
            "messages": messages,
            "employee_id": emp_id,
            "employee_type": emp_type,
            "final_response": ""
        }

        # Run the graph
        result = graph.invoke(initial_state)

        # Extract final response
        final_messages = result.get("messages", [])
        if final_messages:
            last_message = final_messages[-1]

            # Only add AI message if it's not a tool message
            if not isinstance(last_message, ToolMessage):
                chat_history.append(last_message)
                ai_response = getattr(last_message, "content", str(last_message))
            else:
                # Find the last non-tool message (the AI's final response)
                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage):
                        chat_history.append(msg)
                        ai_response = msg.content
                        break
                else:
                    ai_response = "[No response returned by the agent]"
        else:
            ai_response = "[No response returned by the agent]"

        # Keep chat history limited
        chat_history = chat_history[-10:]

        return ai_response, chat_history

    except Exception as e:
        error_msg = f"[Error invoking agent] {str(e)}"
        print(error_msg)
        return error_msg, chat_history


# ----- Helper functions (same as before) -----
def write_employee_type(emp_type):
    global employee_type
    employee_type = emp_type
    print(f"DEBUG: Streamlit_Agent.employee_type set to: {employee_type}")


def write_employee_id(emp_id):
    """Set the employee ID at runtime"""
    global employee_id
    employee_id = emp_id
    print(f"DEBUG: Streamlit_Agent.employee_id set to: {employee_id}")
