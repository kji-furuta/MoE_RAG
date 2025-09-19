# Repository Guidelines

Contributors should treat this repo as a modular RAG platform with shared utilities and FastAPI services. Use the pointers below to stay consistent with existing workflows.

## Project Structure & Module Organization
- `src/` holds Python packages: `rag/` for retrieval and indexing, `training/` for LoRA/DoRA routines, `moe_rag_integration/`, `inference/`, and shared `utils/`.
- `app/` contains the FastAPI entrypoint `app/main_unified.py` plus UI assets.
- `scripts/` provides startup and conversion helpers; `docker/` keeps Dockerfiles and compose configs.
- Tests live under `tests/` as `test_*.py`; data artifacts belong in `data/`, `models/`, `outputs/`, and templates under `templates/`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate && pip install -e .[dev]`: standard local setup.
- `python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload`: launch the API with hot reload.
- `pytest -q` or `pytest -m "not integration"`: run unit suites; skip long integrations when needed.
- `black . && isort . && flake8 src tests`: format and lint before committing.

## Coding Style & Naming Conventions
- Target Python 3.8+, 4-space indentation, UTF-8 files.
- Apply Black (line length 88) and isort (profile "black"); keep imports sorted.
- Modules use `snake_case.py`, classes `CapWords`, functions and variables `snake_case`, constants `UPPER_SNAKE`.

## Testing Guidelines
- Write tests with pytest under `tests/` using `test_*` files and descriptive function names.
- Mark slow or external suites with `@pytest.mark.integration`.
- Prefer fast, deterministic cases; include fixtures for model paths or configs when necessary.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (e.g., `feat: RAG model selection improvements`).
- PRs should describe purpose, scope, test evidence, related issues, and migration notes for config updates.
- Exclude large artifacts; keep `data/`, `models/`, and `outputs/` out of version control unless they are tiny samples.

## Security & Configuration Tips
- Store secrets in `.env` (see `.env.example`); never check credentials into Git.
- Document config defaults in `config/` or `src/rag/config/` when adding options.
- Update Docker and script references if you move GPU or storage paths.

## Agent-Specific Instructions
- Keep diffs minimal and run the formatting + lint stack before submitting.
- Avoid touching unrelated files; ask for guidance if unexpected changes appear in the worktree.
- Document any user-facing behavior change in README or relevant docs.
