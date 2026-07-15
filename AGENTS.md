# LocateAnything-3B

## Agent skills

### Issue tracker

Tickets live in Trello (board: LocateAnything-3B); code lives in a GitHub repo. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root (not yet created; will be added lazily by `/domain-modeling`). See `docs/agents/domain.md`.

### Triage labels

Canonical roles map 1:1 to Trello labels of the same name (`bug`/`enhancement`, `needs-triage`/`needs-info`/`ready-for-agent`/`ready-for-human`/`wontfix`). See `docs/agents/triage-labels.md`.

### Coding standards

Python: `ruff` for lint + format (`pyproject.toml`), type hints required, Pydantic models for all API contracts, thin endpoints. See `docs/CODING_STANDARDS.md`.
