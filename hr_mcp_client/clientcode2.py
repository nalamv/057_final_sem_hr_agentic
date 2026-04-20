import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Updated Imports
from langchain_groq import ChatGroq
from langchain_community.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool


class LLMInterface:
    """Separated LLM Class for Groq Integration"""

    def __init__(self, model_name: str = "llama3-70b-8192"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("Please set GROQ_API_KEY in environment")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=0
        )

    def get_model(self):
        return self.llm


def create_agent(llm, tools, system_prompt):
    """Factory function to create a LangChain agent"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    # create_tool_calling_agent is the modern standard for models that support tools
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


class MCPClient:
    def __init__(self, llm_wrapper: LLMInterface):
        self.llm = llm_wrapper.get_model()
        # Ensure 'mcp.py' exists in your directory
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tools_path = "src/mcp/mcp_server_employee_tools.py"
        _tools_path = os.path.join(PROJECT_ROOT, tools_path)
        self.server_params = StdioServerParameters(
            command="python",
            args=[_tools_path],
        )

    async def _get_tools_from_server(self, session):
        """Discovers tools from MCP server and wraps them for LangChain"""
        mcp_tools_resp = await session.list_tools()
        lc_tools = []

        for tool in mcp_tools_resp.tools:
            # Closure to capture tool name for the async call
            def make_call(t_name):
                async def call_mcp_tool(**kwargs):
                    result = await session.call_tool(t_name, kwargs)
                    return result.content[0].text

                return call_mcp_tool

            lc_tools.append(StructuredTool.from_function(
                coroutine=make_call(tool.name),
                name=tool.name,
                description=tool.description
            ))
        return lc_tools

    async def process_query(self, user_query: str):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Dynamically build tools from the MCP Server
                tools = await self._get_tools_from_server(session)

                # Initialize Agent using the separate class logic
                agent_executor = create_agent(
                    self.llm,
                    tools,
                    "You are a helpful HR assistant. Be concise."
                )

                # Run the agent
                response = await agent_executor.ainvoke({"input": user_query})
                return response["output"]


async def main():
    # Initialize components
    llm_bridge = LLMInterface()  # Uses GROQ_API_KEY from env
    client = MCPClient(llm_bridge)

    # Test Query
    prompt = "Get all employees and then tell me if Alice is registered."
    print(f"User: {prompt}")

    result = await client.process_query(prompt)
    print(f"\nFinal Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())