# Repository Guidelines

## Project Structure & Module Organization
- `src/` — core Python modules
  - `rag/` (query engine, indexing, retrieval), `training/` (LoRA/DoRA, continual), `moe_rag_integration/`, `inference/`, `utils/`.
- `app/` — FastAPI app (`app/main_unified.py`) and UI assets.
- `scripts/` — ops and utilities (start web, model prep, conversions).
- `tests/` — unit/integration tests (`test_*.py`).
- `docker/` — Dockerfile, `docker-compose.yml` for full stack.
- `config/`, `configs/` — runtime and training configs.
- Data and outputs: `data/`, `models/`, `outputs/`, `templates/`.

## Build, Test, and Development Commands
- Setup (local): `python -m venv venv && source venv/bin/activate && pip install -e .[dev]` (or `pip install -r requirements.txt`).
- Lint/Format: `black . && isort . && flake8 src tests`.
- Tests: `pytest -q` (skip heavy tests: `pytest -m "not integration"`).
- Run API (local): `python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload`.
- Docker stack: `cd docker && docker-compose up -d --build` then `bash scripts/start_web_interface.sh`.

## Coding Style & Naming Conventions
- Python 3.8+; Black line length 88, isort profile “black”.
- Indentation 4 spaces; UTF-8; keep functions small and cohesive.
- Naming: modules/files `snake_case.py`, classes `CapWords`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Keep public APIs stable in `src/rag/` and `app/`; add docstrings for new modules.

## Testing Guidelines
- Primary: pytest; some unittest suites exist.
- Location: `tests/` with `test_*.py` per component (e.g., RAG, MoE, deps).
- Marks: long‑running/infra tests use `@pytest.mark.integration`.
- Run locally: `pytest -q`; before PRs: `pytest -q && flake8 && black --check . && isort --check-only .`.

## Commit & Pull Request Guidelines
- Use Conventional Commits seen in history: `feat: …`, `fix: …`, `docs: …`, `chore: …` (EN/JP OK). Example: `feat: RAG model selection improvements`.
- PRs must include: purpose, scope, test plan/output, screenshots or logs for UI/API, related issues, and migration notes if configs change.
- Do not commit large artifacts; keep `data/`, `models/`, `outputs/` out of Git unless tiny samples.

## Security & Configuration Tips
- Secrets via `.env` (see `.env.example`); never commit keys.
- Config lives in `config/` and `src/rag/config/`; document defaults/overrides.
- GPU/Docker paths are referenced by scripts—avoid renaming without updating `scripts/` and `docker/`.

## Agent-Specific Notes
- Prefer minimal, focused diffs; follow Black/isort.
- Avoid breaking public endpoints or path conventions without updating docs/tests.
- Use `rg` for search; keep changes within scope and update README when user‑facing behavior changes.

