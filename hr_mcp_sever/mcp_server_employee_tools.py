from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("EmployeeManager")
BASE_URL = "http://localhost:9000"

@mcp.tool()
async def register_employee(name: str, email: str, department: str, manager_id: int) -> str:
    """Registers a new employee in the system."""
    payload = {"name": name, "email": email, "department": department, "manager_id": manager_id}
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{BASE_URL}/employee/register", json=payload)
        return res.text

@mcp.tool()
async def apply_leave(employee_id: int, leave_type: str, start_date: str, end_date: str, reason: str) -> str:
    """Submits a leave request for an employee."""
    payload = {
        "employee_id": employee_id, "leave_type": leave_type,
        "start_date": start_date, "end_date": end_date, "reason": reason
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{BASE_URL}/employee/apply-leave", json=payload)
        return res.text

@mcp.tool()
async def get_employee_leaves(emp_id: int) -> str:
    """Retrieves all leave records for a specific employee ID."""
    async with httpx.AsyncClient() as client:
        print(f"*******Get Employee {emp_id} Leave Records")
        res = await client.get(f"{BASE_URL}/employee/{emp_id}/leaves")
        return res.text

@mcp.tool()
async def list_all_employees() -> str:
    """Fetches a list of all employees."""
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{BASE_URL}/employee/all")
        return res.text

if __name__ == "__main__":
    mcp.run()