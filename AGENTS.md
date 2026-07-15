# LocateAnything-3B

## Agent skills

### Issue tracker

Tickets live in Trello (board: LocateAnything-3B); code lives in a GitHub repo. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root (not yet created; will be added lazily by `/domain-modeling`). See `docs/agents/domain.md`.

### Coding standards

Python: `ruff` for lint + format (`pyproject.toml`), type hints required, Pydantic models for all API contracts, thin endpoints. See `docs/CODING_STANDARDS.md`.
