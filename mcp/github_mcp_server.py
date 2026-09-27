"""Launches the official GitHub MCP server (Docker) for this project.

Registered in .mcp.json as the "github" server. Requires Docker and a
token, read from (in order): the GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_PAT
environment variable already set for this user, or GITHUB_PERSONAL_ACCESS_TOKEN
in .env. Create one at https://github.com/settings/personal-access-tokens
with the scopes you need (e.g. repo, read:org). The token is forwarded to the
container by reference (-e NAME with no value) so it never appears in
argv/process listings.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TOKEN_ENV_VAR = next(
    (name for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_PAT") if os.getenv(name)),
    None,
)

if not TOKEN_ENV_VAR:
    print(
        "No GitHub token found. Set GITHUB_PERSONAL_ACCESS_TOKEN or GITHUB_PAT "
        "for this user, or add GITHUB_PERSONAL_ACCESS_TOKEN to .env, before running.",
        file=sys.stderr,
    )
    sys.exit(1)

# github-mcp-server reads its token from GITHUB_PERSONAL_ACCESS_TOKEN specifically,
# so mirror whichever source we found into that name for the container.
os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = os.environ[TOKEN_ENV_VAR]

result = subprocess.run(
    [
        "docker",
        "run",
        "-i",
        "--rm",
        "--name",
        "giri",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server",
        "stdio",
    ]
)
sys.exit(result.returncode)
