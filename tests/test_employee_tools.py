from services.employee_service import EmployeeApiUnavailableError, EmployeeNotFoundError
from tools.employee_tools import make_employee_tools


class FakeEmployeeService:
    def __init__(self, employees=None, raise_error=None):
        self.employees = employees or []
        self.raise_error = raise_error

    def get_employee(self, employee_id: int) -> dict:
        if self.raise_error:
            raise self.raise_error
        for employee in self.employees:
            if employee["id"] == employee_id:
                return employee
        raise EmployeeNotFoundError(f"Employee not found with id: {employee_id}")

    def search_employees(self, name: str) -> list:
        if self.raise_error:
            raise self.raise_error
        name_lower = name.lower()
        return [
            e
            for e in self.employees
            if name_lower in e["firstName"].lower() or name_lower in e["lastName"].lower()
        ]

    def list_employees(self) -> list:
        if self.raise_error:
            raise self.raise_error
        return self.employees


def make_tools(service):
    tools = make_employee_tools(service)
    return {tool.name: tool for tool in tools}


def test_get_employee_rejects_non_positive_id():
    tools = make_tools(FakeEmployeeService())

    result = tools["get_employee"].invoke({"employee_id": 0})

    assert "Invalid employee_id" in result


def test_get_employee_returns_not_found_message():
    tools = make_tools(FakeEmployeeService(employees=[]))

    result = tools["get_employee"].invoke({"employee_id": 999})

    assert "No employee found with id 999" in result


def test_get_employee_returns_formatted_employee():
    employee = {
        "id": 1,
        "firstName": "Ada",
        "lastName": "Lovelace",
        "email": "ada@example.com",
        "department": "Engineering",
        "salary": 100000,
    }
    tools = make_tools(FakeEmployeeService(employees=[employee]))

    result = tools["get_employee"].invoke({"employee_id": 1})

    assert "Ada Lovelace" in result
    assert "Engineering" in result


def test_get_employee_reports_unavailable_service_without_raising():
    tools = make_tools(
        FakeEmployeeService(raise_error=EmployeeApiUnavailableError("Employee API is unreachable."))
    )

    result = tools["get_employee"].invoke({"employee_id": 1})

    assert "unavailable" in result.lower()


def test_search_employee_rejects_blank_name():
    tools = make_tools(FakeEmployeeService())

    result = tools["search_employee"].invoke({"name": "   "})

    assert "Invalid search" in result


def test_search_employee_reports_no_matches():
    tools = make_tools(FakeEmployeeService(employees=[]))

    result = tools["search_employee"].invoke({"name": "Nobody"})

    assert "No employees found matching 'Nobody'" in result


def test_list_employees_reports_empty():
    tools = make_tools(FakeEmployeeService(employees=[]))

    result = tools["list_employees"].invoke({})

    assert "No employees found" in result


def test_list_employees_filters_by_department_locally():
    employees = [
        {"id": 1, "firstName": "Ada", "lastName": "Lovelace", "email": "a@x.com", "department": "Engineering", "salary": 1},
        {"id": 2, "firstName": "Bob", "lastName": "Smith", "email": "b@x.com", "department": "Sales", "salary": 1},
    ]
    tools = make_tools(FakeEmployeeService(employees=employees))

    result = tools["list_employees"].invoke({"department": "Sales"})

    assert "Bob Smith" in result
    assert "Ada Lovelace" not in result
