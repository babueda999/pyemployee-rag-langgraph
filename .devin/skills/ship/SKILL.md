---
name: ship
description: Commit changes to dev, push to GitHub, open a PR to main, and merge it. Use when the user asks to push, ship, or open a PR for completed work.
---

# Ship

Repo: https://github.com/babueda999/pyemployee-rag-langgraph
Flow: `dev` -> PR -> `main` -> auto-sync back to `dev`.

1. `git status` / `git diff` / `git log --oneline -5` — review what changed
   and match commit style. Do not stage `nul` (stray Windows artifact) or
   anything gitignored.
2. Run the `verify` skill's pytest command before committing.
3. Commit with a "why"-focused message and the Devin trailer
   (`Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>`).
4. `git pull --no-rebase origin dev` if behind, resolve, then
   `git push origin dev`.
5. `gh pr create --base main --head dev` (no PR template exists).
6. Merge with `gh pr merge` when the user approves or asks you to merge.

Rules:

- NEVER hand-merge `main` into `dev` — `.github/workflows/sync-dev-with-main.yml`
  does it automatically on every `main` push.
- If `gh` times out locally, the GitHub MCP server tools still work
  (Docker networking differs).
- `main` has no branch protection yet — merges are not gated on approval.
