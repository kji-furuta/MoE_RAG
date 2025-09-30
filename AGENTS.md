# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts core packages: `rag/` for retrieval, `training/` for LoRA/DoRA and continual learning, `moe_rag_integration/`, shared `inference/`, and `utils/` helpers.
- `app/` serves the FastAPI entrypoint (`app/main_unified.py`) and UI assets; `scripts/` centralizes launchers and model prep tasks; `docker/` provides deployment manifests.
- Config lives in `config/` and `configs/`; generated artifacts belong in `data/`, `models/`, `outputs/`, and `templates/`.
- Tests reside in `tests/` using `test_*.py`; heavier demos land in `examples/` and `notebooks/`.

## Build, Test, and Development Commands
- Bootstrap: `python -m venv venv && source venv/bin/activate && pip install -e .[dev]` (or `pip install -r requirements.txt`).
- Quality gates: `black . && isort . && flake8 src tests` keeps formatting and linting aligned with CI.
- Unit suite: `pytest -q`; skip integration jobs with `pytest -m "not integration"`.
- Local API: `python -m uvicorn app.main_unified:app --host 0.0.0.0 --port 8050 --reload`.
- Full stack: `cd docker && docker-compose up -d --build` then `bash scripts/start_web_interface.sh`.

## Coding Style & Naming Conventions
- Target Python 3.8+, Black’s 88-char limit, isort profile "black", and 4-space indentation.
- Use `snake_case` for modules and functions, `CapWords` for classes, and `UPPER_SNAKE` for constants.
- Keep public interfaces in `src/rag/` and `app/` stable; add tight docstrings for new modules or intricate logic.

## Testing Guidelines
- Rely on pytest; mirror package structure when adding `test_*.py` files under `tests/`.
- Run `pytest -q` before pushes; for release branches include `pytest -q && flake8 && black --check . && isort --check-only .`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`). Example: `feat: improve hybrid retriever scoring`.
- PRs must outline purpose, scope, and linked issues; attach logs or screenshots for UI/API changes and note config or data migrations.
- Exclude large weights or datasets—only minimal samples belong in Git.

## Security & Configuration Tips
- Load secrets through `.env` (see `.env.example`); never hard-code credentials.
- Document overrides when editing `config/` or `src/rag/config/` and update related scripts or Docker references.
- Treat `data/`, `models/`, and `outputs/` as runtime storage; purge sensitive artifacts before sharing.

## Agent Playbook
- Prefer targeted diffs, rely on `rg` for search, and run quick smoke checks after edits.
- Surface blocking issues immediately; avoid resetting user changes unless instructed.
