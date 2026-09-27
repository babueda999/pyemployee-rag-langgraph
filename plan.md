# Git / GitHub Workflow

Repo: https://github.com/babueda999/pyemployee-rag-langgraph

## Branching model

- `main` — protected target branch. Receives changes only via merged PRs from `dev`.
- `dev` — working branch. Commit and push new work here.

## Flow

1. Commit changes to `dev`, push to `origin/dev`.
2. Open a PR from `dev` into `main`.
3. Review and merge the PR.
4. `.github/workflows/sync-dev-with-main.yml` runs automatically on every push to
   `main`, merges `main` back into `dev`, and pushes — so `dev` never drifts from
   `main` after a merge.

## Auth

`GITHUB_PAT` in the local environment is the `employee-services-mcp` fine-grained
token (name predates this repo). Currently scoped to:
- `babueda999/employee-services`
- `babueda999/pyemployee-rag-langgraph`

Repository permissions granted: Contents (read/write), Pull requests (read/write),
Workflows (read/write), Metadata (read-only, required).

## Open items

- [ ] Branch protection on `main` (require PR before merge, block direct pushes) —
      needs the `Administration` permission added to the token first.
- [ ] Repo visibility is currently **public**; flip to private if that was intended.
- [ ] A second token, `GithubToken`, was scoped to this repo with Contents
      read/write while debugging the above but is unused by anything — revert to
      no access or repurpose.
