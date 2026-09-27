"""Launches the mcp-atlassian MCP server (Docker) for this project's Jira board.

Registered in .mcp.json as the "jira" server. Requires Docker and, for this
user, already-set environment variables JIRA_SITE_URL, JIRA_EMAIL, and
JIRA_API_TOKEN (falls back to the mcp-atlassian's own JIRA_URL/JIRA_USERNAME
names, or .env, if those are set instead). Values are forwarded to the
container by reference (-e NAME with no value) so they never appear in
argv/process listings.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _first_set(*names: str) -> str | None:
    return next((name for name in names if os.getenv(name)), None)


url_var = _first_set("JIRA_URL", "JIRA_SITE_URL")
username_var = _first_set("JIRA_USERNAME", "JIRA_EMAIL")
token_var = _first_set("JIRA_API_TOKEN", "JIRA_TOKEN")

missing = [
    label
    for label, var in (("URL", url_var), ("username/email", username_var), ("API token", token_var))
    if not var
]
if missing:
    print(
        f"Missing Jira {', '.join(missing)}. Set JIRA_SITE_URL/JIRA_URL, "
        "JIRA_EMAIL/JIRA_USERNAME, and JIRA_API_TOKEN for this user, or add them to .env.",
        file=sys.stderr,
    )
    sys.exit(1)

# mcp-atlassian reads JIRA_URL / JIRA_USERNAME / JIRA_API_TOKEN specifically,
# so mirror whichever source names we found into those for the container.
os.environ["JIRA_URL"] = os.environ[url_var]
os.environ["JIRA_USERNAME"] = os.environ[username_var]
os.environ["JIRA_API_TOKEN"] = os.environ[token_var]

result = subprocess.run(
    [
        "docker",
        "run",
        "-i",
        "--rm",
        "--name",
        "sesha",
        "-e",
        "JIRA_URL",
        "-e",
        "JIRA_USERNAME",
        "-e",
        "JIRA_API_TOKEN",
        "ghcr.io/sooperset/mcp-atlassian:latest",
        "--transport",
        "stdio",
    ]
)
sys.exit(result.returncode)
