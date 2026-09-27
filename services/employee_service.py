"""Thin client for the existing Spring Boot Employee REST API.

Only ever talks to a fixed, configured base URL plus known path templates
(/api/employees, /api/employees/{id}, /api/employees/search) — never an
arbitrary caller-supplied URL.
"""

import os

import requests


class EmployeeApiError(Exception):
    """Base error for any failure talking to the Employee API."""


class EmployeeNotFoundError(EmployeeApiError):
    """Raised when the Employee API returns 404 for a given id."""


class EmployeeApiUnavailableError(EmployeeApiError):
    """Raised when the Employee API is unreachable or times out."""


class EmployeeServiceClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or os.getenv("EMPLOYEE_API_BASE_URL", "http://localhost:8080")).rstrip("/")
        self.timeout = timeout or float(os.getenv("EMPLOYEE_API_TIMEOUT_SECONDS", "5"))
        self._session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            return self._session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            raise EmployeeApiUnavailableError(
                f"Employee API timed out after {self.timeout}s calling {path}."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise EmployeeApiUnavailableError(
                f"Employee API is unreachable at {self.base_url}."
            ) from exc

    def get_employee(self, employee_id: int) -> dict:
        response = self._get(f"/api/employees/{int(employee_id)}")

        if response.status_code == 404:
            raise EmployeeNotFoundError(f"Employee not found with id: {employee_id}")
        if not response.ok:
            raise EmployeeApiError(
                f"Employee API returned {response.status_code} fetching employee {employee_id}."
            )
        return response.json()

    def search_employees(self, name: str) -> list[dict]:
        response = self._get("/api/employees/search", params={"name": name})

        if not response.ok:
            raise EmployeeApiError(
                f"Employee API returned {response.status_code} searching for '{name}'."
            )
        return response.json()

    def list_employees(self) -> list[dict]:
        response = self._get("/api/employees")

        if not response.ok:
            raise EmployeeApiError(f"Employee API returned {response.status_code} listing employees.")
        return response.json()


def get_employee_service() -> EmployeeServiceClient:
    return EmployeeServiceClient()
