"""LangChain tools exposing the Spring Boot Employee API to the agent.

Each tool validates its own inputs and turns service-layer errors into a
plain string result instead of raising, so the agent can report that
information is unavailable rather than crashing or guessing.
"""

from langchain_core.tools import tool

from services.employee_service import (
    EmployeeApiError,
    EmployeeApiUnavailableError,
    EmployeeNotFoundError,
    EmployeeServiceClient,
)

MAX_NAME_LENGTH = 200


def _format_employee(employee: dict) -> str:
    return (
        f"id={employee.get('id')}, "
        f"name={employee.get('firstName')} {employee.get('lastName')}, "
        f"email={employee.get('email')}, "
        f"department={employee.get('department')}, "
        f"salary={employee.get('salary')}, "
        f"remote_work_eligible={employee.get('remoteWorkEligible')}"
    )


def _format_employees(employees: list[dict]) -> str:
    return "\n".join(_format_employee(employee) for employee in employees)


def make_employee_tools(service: EmployeeServiceClient) -> list:
    @tool
    def get_employee(employee_id: int) -> str:
        """Look up a single employee by their numeric id. Returns their
        name, email, department, and salary, or says they were not found."""
        if not isinstance(employee_id, int) or isinstance(employee_id, bool) or employee_id <= 0:
            return "Invalid employee_id: must be a positive integer."

        try:
            employee = service.get_employee(employee_id)
        except EmployeeNotFoundError:
            return f"No employee found with id {employee_id}."
        except EmployeeApiUnavailableError as exc:
            return f"Employee service is currently unavailable: {exc}"
        except EmployeeApiError as exc:
            return f"Employee service error: {exc}"

        return _format_employee(employee)

    @tool
    def search_employee(name: str) -> str:
        """Search employees by first or last name (case-insensitive,
        partial match). Returns matching employees or says none were found."""
        name = (name or "").strip()
        if not name:
            return "Invalid search: name must not be empty."
        if len(name) > MAX_NAME_LENGTH:
            return f"Invalid search: name must be {MAX_NAME_LENGTH} characters or fewer."

        try:
            employees = service.search_employees(name)
        except EmployeeApiUnavailableError as exc:
            return f"Employee service is currently unavailable: {exc}"
        except EmployeeApiError as exc:
            return f"Employee service error: {exc}"

        if not employees:
            return f"No employees found matching '{name}'."
        return _format_employees(employees)

    @tool
    def list_employees(department: str | None = None) -> str:
        """List all employees. Optionally filter by department (filtering
        is done locally since the API has no server-side department
        filter). Says when none are found."""
        try:
            employees = service.list_employees()
        except EmployeeApiUnavailableError as exc:
            return f"Employee service is currently unavailable: {exc}"
        except EmployeeApiError as exc:
            return f"Employee service error: {exc}"

        if department:
            department_lower = department.strip().lower()
            employees = [
                employee
                for employee in employees
                if (employee.get("department") or "").lower() == department_lower
            ]

        if not employees:
            scope = f" in department '{department}'" if department else ""
            return f"No employees found{scope}."
        return _format_employees(employees)

    return [get_employee, search_employee, list_employees]
