import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool


class EmployeeAssistant:
    def __init__(self, llm_wrapper: LLMInterface):
        self.llm = llm_wrapper.get_model()
        self.server_params = StdioServerParameters(
            command="python",
            args=["mcp.py"],  # Ensure your server script is named this
        )

    def _convert_mcp_to_langchain_tools(self, mcp_tools, session):
        """Converts MCP tools into LangChain StructuredTool objects."""
        lc_tools = []
        for tool in mcp_tools:
            # We create a wrapper function that calls the MCP session
            def create_tool_func(t_name):
                async def tool_func(**kwargs):
                    result = await session.call_tool(t_name, kwargs)
                    return result.content[0].text

                return tool_func

            lc_tools.append(StructuredTool.from_function(
                coroutine=create_tool_func(tool.name),
                name=tool.name,
                description=tool.description
            ))
        return lc_tools

    async def chat(self, user_input: str):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. Get tools from MCP Server
                mcp_tools_resp = await session.list_tools()
                tools = self._convert_mcp_to_langchain_tools(mcp_tools_resp.tools, session)

                # 2. Define the Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Be concise and accurate."),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])

                # 3. Create the LangChain Agent
                # Using tool_calling_agent which is ideal for Groq/OpenAI models
                agent = create_tool_calling_agent(self.llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 4. Execute
                response = await agent_executor.ainvoke({"input": user_input})
                return response["output"]


async def main():
    # Usage
    llm_bridge = LLMInterface()
    assistant = EmployeeAssistant(llm_bridge)

    query = "Who are all the employees? Also, register 'Alice' in HR (dept) with manager 5."
    print(f"User: {query}")

    result = await assistant.chat(query)
    print(f"Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())