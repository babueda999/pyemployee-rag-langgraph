from unittest.mock import patch

import requests

from services.employee_service import (
    EmployeeApiError,
    EmployeeApiUnavailableError,
    EmployeeNotFoundError,
    EmployeeServiceClient,
)


class FakeResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data

    def json(self):
        return self._json_data


def make_client() -> EmployeeServiceClient:
    return EmployeeServiceClient(base_url="http://localhost:8080", timeout=1)


def test_get_employee_returns_employee_on_success():
    client = make_client()
    employee = {"id": 1, "firstName": "Ada", "lastName": "Lovelace", "department": "Engineering"}

    with patch.object(requests.Session, "get", return_value=FakeResponse(200, employee)):
        result = client.get_employee(1)

    assert result == employee


def test_get_employee_raises_not_found_on_404():
    client = make_client()

    with patch.object(requests.Session, "get", return_value=FakeResponse(404)):
        try:
            client.get_employee(999)
            assert False, "expected EmployeeNotFoundError"
        except EmployeeNotFoundError:
            pass


def test_get_employee_raises_api_error_on_server_error():
    client = make_client()

    with patch.object(requests.Session, "get", return_value=FakeResponse(500)):
        try:
            client.get_employee(1)
            assert False, "expected EmployeeApiError"
        except EmployeeApiError:
            pass


def test_get_employee_raises_unavailable_on_timeout():
    client = make_client()

    with patch.object(requests.Session, "get", side_effect=requests.exceptions.Timeout()):
        try:
            client.get_employee(1)
            assert False, "expected EmployeeApiUnavailableError"
        except EmployeeApiUnavailableError:
            pass


def test_get_employee_raises_unavailable_on_connection_error():
    client = make_client()

    with patch.object(requests.Session, "get", side_effect=requests.exceptions.ConnectionError()):
        try:
            client.get_employee(1)
            assert False, "expected EmployeeApiUnavailableError"
        except EmployeeApiUnavailableError:
            pass


def test_search_employees_returns_matches():
    client = make_client()
    matches = [{"id": 1, "firstName": "Ada", "lastName": "Lovelace"}]

    with patch.object(requests.Session, "get", return_value=FakeResponse(200, matches)):
        result = client.search_employees("Ada")

    assert result == matches


def test_list_employees_returns_all():
    client = make_client()
    employees = [{"id": 1}, {"id": 2}]

    with patch.object(requests.Session, "get", return_value=FakeResponse(200, employees)):
        result = client.list_employees()

    assert result == employees
