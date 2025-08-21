# Development Runbook

## Repository layout
- `src/imdb_recommender` – library code and CLIs
- `tests/unit` – fast tests for individual modules
- `tests/integration` – end-to-end tests exercising multiple modules
- `scripts/training` – model training jobs
- `scripts/analysis` – exploratory notebooks and reports
- `scripts/diagnostics` – profiling and debugging utilities
- `scripts/selenium` – Selenium automation helpers (create as needed)
- `docs` – project documentation
- `data/raw` – unversioned IMDb exports (ignored)
- `data/processed` – derived datasets (ignored)
- `results` – experiment outputs and checkpoints (ignored)
- `.github/workflows` – CI configuration

## Quick start
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"
pytest -q
ruff check src tests
black --check src tests
git config core.hooksPath .githooks
```
Copy `.env.example` to `.env` and fill in paths or credentials as needed. The
application loads environment variables via `python-dotenv` when scripts run.

## Commands
- `ruff check src tests` – style and static checks
- `black --check src tests` – ensure formatting
- `pytest --cov=imdb_recommender --cov-report=xml` – run tests with coverage
- Optional changed-lines coverage using diff-cover:
```
pytest --cov=imdb_recommender --cov-report=xml
pip install diff-cover
diff-cover coverage.xml --compare-branch=origin/main --fail-under=85
```

## Data policy
Never commit real datasets. Use fixtures under `tests/fixtures/data/*`.
```
# Use synthetic samples; pass paths explicitly
python scripts/training/run_elasticnet_cv.py \
  --ratings-file tests/fixtures/data/ratings_sample.csv \
  --watchlist-file tests/fixtures/data/watchlist_sample.csv \
  --topk 10
```
Pass custom paths via CLI flags (`--ratings-file`, `--watchlist-file`) or via
environment variables `RATINGS_CSV_PATH` and `WATCHLIST_PATH` in `.env`.
Deprecated aliases `--ratings` and `--watchlist` remain available but emit a warning.

## Development workflow
- Branch names: `feature/description` or `bugfix/description`
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- Add fixtures under `tests/fixtures` for new data samples
- Run unit vs integration tests separately when iterating:
```
pytest tests/unit -q
pytest tests/integration -q
```

## Quality gates
- `ruff` and `black` report zero errors
- Test coverage ≥ 85%
- Run `bandit` or other security scans if enabled
- Update docs and changelog for user-facing changes

## Continuous Integration
GitHub Actions (`.github/workflows/ci.yml`) runs:
- `pytest tests/ -q` on Python 3.12
- `ruff check .` and `black --check --diff .` (warnings only)
Reproduce locally with the commands in this runbook.

## Troubleshooting
- **Missing extras**: ensure dev/test extras installed with `pip install -e ".[dev,test]"`
- **Path errors**: supply `--ratings-file` and `--watchlist-file` or set env vars
- **Fixture not found**: add sample data under `tests/fixtures/data`
- **Non‑determinism**: set `RANDOM_SEED` in `.env` or CLI to reproduce results
