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

BASE_URL = "http://127.0.0.1:9000"

app = Server("employee-api-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="register_employee",
            description="Register a new employee in the system",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the employee"},
                    "email": {"type": "string", "description": "Email address of the employee"},
                    "department": {"type": "string", "description": "Department the employee belongs to"},
                    "manager_id": {"type": "integer", "description": "ID of the employee's manager (use 0 if no manager)"},
                },
                "required": ["name", "email", "department", "manager_id"],
            },
        ),
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
        ),
        types.Tool(
            name="get_all_employees",
            description="Retrieve the list of all employees registered in the system",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    async with httpx.AsyncClient() as client:
        try:
            if name == "register_employee":
                response = await client.post(
                    f"{BASE_URL}/employee/register",
                    json={
                        "name": arguments["name"],
                        "email": arguments["email"],
                        "department": arguments["department"],
                        "manager_id": arguments["manager_id"],
                    },
                )
                result = response.json()

            elif name == "apply_leave":
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

            elif name == "get_all_employees":
                response = await client.get(f"{BASE_URL}/employee/all")
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