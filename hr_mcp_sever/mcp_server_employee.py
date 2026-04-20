"""
MCP Server - Employee API Tools
Exposes REST API endpoints as MCP tools for LLM consumption.
"""

import asyncio
import json
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

BASE_URL = "http://127.0.0.1:8000"

app = Server("employee-role-api-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="apply_leave",
            description="Apply for a leave for an employee",
            inputSchema={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "integer", "description": "ID of the employee applying for leave"},
                    "leave_type": {"type": "string", "description": "Type of leave (e.g., sick, casual, annual)"},
                    "start_date": {"type": "string", "description": "Start date of leave in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date of leave in YYYY-MM-DD format"},
                    "reason": {"type": "string", "description": "Reason for the leave"},
                },
                "required": ["employee_id", "leave_type", "start_date", "end_date", "reason"],
            },
        ),
        types.Tool(
            name="get_employee_leaves",
            description="Get all leave records for a specific employee by their ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "employee_id": {"type": "integer", "description": "The ID of the employee"},
                },
                "required": ["employee_id"],
            },
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    async with httpx.AsyncClient() as client:
        try:
           if name == "apply_leave":
                response = await client.post(
                    f"{BASE_URL}/employee/apply-leave",
                    json={
                        "employee_id": arguments["employee_id"],
                        "leave_type": arguments["leave_type"],
                        "start_date": arguments["start_date"],
                        "end_date": arguments["end_date"],
                        "reason": arguments["reason"],
                    },
                )
                result = response.json()

            elif name == "get_employee_leaves":
                emp_id = arguments["employee_id"]
                response = await client.get(f"{BASE_URL}/employee/{emp_id}/leaves")
                result = response.json()

            else:
                result = {"error": f"Unknown tool: {name}"}

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        except httpx.ConnectError:
            error_msg = f"Connection error: Could not connect to {BASE_URL}. Make sure the Employee API server is running."
            return [types.TextContent(type="text", text=json.dumps({"error": error_msg}))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())