"""
MCP Client - Employee API Chat Interface
Uses ChatGroq (LangChain) as the LLM to process natural language
and route tool calls through the MCP server.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-20b"  # or mixtral-8x7b-32768, gemma2-9b-it, etc.

SYSTEM_PROMPT = """You are a helpful HR assistant that manages employee records and leave requests.
You have access to the following tools:
- register_employee: Register a new employee
- apply_leave: Apply for a leave on behalf of an employee
- get_employee_leaves: View leave records for a specific employee
- get_all_employees: List all employees in the system

When the user asks you to perform any of these actions, use the appropriate tool.
Always be friendly and confirm actions after they are completed.
If you need information to complete a task (like an employee ID), ask the user for it.
"""


class MCPGroqClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=0,
        )
        self.tools_map: dict = {}          # name -> MCP tool schema
        self.langchain_tools: list = []    # OpenAI-format tool schemas for LLM
        self.chat_history: list = [SystemMessage(content=SYSTEM_PROMPT)]

    # ── Connect to MCP server via stdio ──────────────────────────────────
    async def connect(self, server_script: str = "mcp_server.py"):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()
        await self._load_tools()
        print(f"✅  Connected to MCP server. Available tools: {list(self.tools_map.keys())}\n")

    # ── Fetch tool schemas from MCP server ───────────────────────────────
    async def _load_tools(self):
        response = await self.session.list_tools()
        for t in response.tools:
            self.tools_map[t.name] = t
            # Convert MCP schema → OpenAI function-calling format
            self.langchain_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            })

    # ── Call a tool on the MCP server ────────────────────────────────────
    async def _call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments)
        # Collect all text content blocks
        texts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(texts) if texts else "No result returned."

    # ── Single turn: send message, handle tool calls, return reply ───────
    async def chat(self, user_input: str) -> str:
        self.chat_history.append(HumanMessage(content=user_input))

        llm_with_tools = self.llm.bind_tools(self.langchain_tools)

        while True:
            response = await llm_with_tools.ainvoke(self.chat_history)
            self.chat_history.append(response)

            # No tool calls → plain text reply
            if not response.tool_calls:
                return response.content

            # Process each tool call
            tool_results = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = tc["id"]

                print(f"  🔧  Calling tool: {tool_name}")
                print(f"      Arguments : {json.dumps(tool_args, indent=6)}")

                result_text = await self._call_tool(tool_name, tool_args)
                print(f"      Result    : {result_text[:200]}{'...' if len(result_text) > 200 else ''}\n")

                tool_results.append(
                    ToolMessage(content=result_text, tool_call_id=tool_call_id)
                )

            self.chat_history.extend(tool_results)
            # Loop back so the LLM can see the tool results and reply

    # ── Interactive REPL ─────────────────────────────────────────────────
    async def run_chat_loop(self):
        print("=" * 60)
        print("  Employee Management Assistant  (type 'exit' to quit)")
        print("=" * 60)
        print("Examples:")
        print("  • Register a new employee named Alice in Engineering")
        print("  • Apply sick leave for employee 3 from 2026-03-10 to 2026-03-12")
        print("  • Show all leaves for employee 5")
        print("  • List all employees")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Goodbye!")
                break

            try:
                reply = await self.chat(user_input)
                print(f"\nAssistant: {reply}\n")
            except Exception as e:
                print(f"\n[Error] {e}\n")

    async def close(self):
        await self.exit_stack.aclose()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
async def main():
    client = MCPGroqClient()
    try:
        await client.connect("mcp_server.py")
        await client.run_chat_loop()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())