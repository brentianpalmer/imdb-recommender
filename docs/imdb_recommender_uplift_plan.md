# IMDb Recommender Uplift Plan

Goal
- Deliver a minimal, maintainable system that:
  - Ingests two files: ratings CSV and watchlist CSV/XLSX
  - Generates recommendations from the watchlist
  - Provides built-in train/test evaluation with optional CV diagnostics
  - Keeps hyperparameter tuning as a dev-only workflow using an internal API
- Reduce sprawl in docs/results/scripts and remove duplication
- Reuse current working code as much as possible; refactor in place, not a rewrite

Scope (reuse-first)
- Keep only two recommenders under a single interface (existing files):
  - imdb_recommender.recommender_svd.SVDAutoRecommender
  - imdb_recommender.recommender_elasticnet.ElasticNetRecommender
- Reuse and minimally adjust:
  - Data IO: src/imdb_recommender/data_io.py
  - CLI: src/imdb_recommender/cli.py
  - Ranker/blending: src/imdb_recommender/ranker.py
  - Base interface: src/imdb_recommender/recommender_base.py
  - Schemas/DTOs: src/imdb_recommender/schemas.py
- Provide new, small modules for evaluation and internal model selection only where missing

Current State – Issues
- Duplicative scripts and oversized CLI (overlapping commands)
- Inconsistent score scales across models
- ElasticNet feature pipeline partially implemented; scaling not wired
- Sprawling results/docs; hard to find the signal
- No automated metric guardrails; risk of regressions

Target End State
- CLI commands:
  - recommend: single entry-point for recommendations with content filtering
    - --content-type movies|tv|all (default: all)
  - eval: offline diagnostics (train/test) with optional CV summary via --cv N
- Hyperparameter tuning: dev-only script(s) using shared internal API; outputs small JSON/CSV artifacts that can feed into model defaults
- Single data ingress path that normalizes input columns once
- Evaluation module (RMSE/MAE and ranking@K), reusable by both models
- Internal-only model_selection module (CV and grid search)
- Consistent score scale across models (normalize to 0–1 before blending)
- Trimmed docs/results; archived legacy artifacts
- Test suite with functional and performance guardrails (protect ElasticNet RMSE baseline ~1.386 ± 0.095)

Content-type filtering (TV vs Movies)
- Implement once in data_io/utils and reuse in CLI:
  - movies: title_type in {"movie", "tvMovie"}
  - tv: title_type in {"tvSeries", "tvMiniSeries", "tvSpecial"} (exclude episodes by default to avoid noisy recs; make set configurable)
- CLI: imdbrec recommend --content-type movies|tv|all
- Keep legacy aliases (top-watchlist-movies, top-watchlist-tv) as thin wrappers over recommend

Phased Refactor Plan

Phase 1 — Core API and CLI simplification (reuse-first)
1) CLI consolidation
- Keep only:
  - imdbrec recommend --ratings-file ... --watchlist-file ... --model svd|elasticnet --topk 10 --content-type movies|tv|all
  - imdbrec eval --ratings-file ... --watchlist-file ... --model ... [--cv 5] [--out results/metrics/run.json]
- Deprecate/alias previous “top-watchlist-*” commands to call recommend with --content-type movies|tv
- Minimal, in-place edits to src/imdb_recommender/cli.py (reuse existing option parsing)

2) Data normalization
- Centralize column mapping/cleaning in data_io; do not duplicate in models
- Standard columns: imdb_const, my_rating, title_type, year, imdb_rating, num_votes, genres, directors, release_date, rated_at
- Implement content-type filtering helper in data_io or utils; import in CLI
- Modify in place: src/imdb_recommender/data_io.py

3) Model interface hardening
- Keep current class names and constructor signatures where possible
- Align score semantics (normalize to 0–1) either in model outputs or in ranker
- Only surgical changes in:
  - src/imdb_recommender/recommender_svd.py
  - src/imdb_recommender/recommender_elasticnet.py
  - src/imdb_recommender/ranker.py

Phase 2 — Evaluation and Diagnostics
4) Evaluation module
- Train/test split (random or time-based) with metrics: RMSE, MAE; ranking metrics precision@K/NDCG@K where applicable
- CLI: imdbrec eval ... [--cv N] prints concise table and writes JSON
- API: evaluate_model(dataset, model_name, params, split="random", metrics=[...], cv=None) -> dict
- New file (small, focused): src/imdb_recommender/evaluation.py

5) Score calibration policy
- Normalize model outputs to 0–1 for consistent ranking/blending
- Implement in ranker or return normalized scores from models; document the choice
- Minimal edits: src/imdb_recommender/ranker.py and, if needed, _scale_predictions in ElasticNet

Phase 3 — Model Selection (Internal-only)
6) Internal API and dev script (no CLI “tune”)
- src/imdb_recommender/model_selection.py:
  - cross_validate(dataset, model_name, params, cv) -> DataFrame
  - grid_search(dataset, model_name, param_grid, cv) -> best_params, cv_results
- scripts/diagnostics/tune.py:
  - Runs grid search, writes:
    - results/tuning/{model}_best_params.json
    - results/tuning/{model}_cv_results.csv
- CLI remains discoverable and small; tuning is a dev workflow

Phase 4 — ElasticNet pipeline cleanup (non-rewrite)
7) Finish and simplify feature pipeline
- Wire or remove dead code with minimal diff:
  - Complete _engineer_features, _filter_unreleased_and_insufficient_metadata, _scale_predictions
  - If scaling is kept, do it via sklearn Pipeline inside CV/train folds to avoid leakage
  - Otherwise, rely on bounded, engineered features (log_votes, clamped year, one-hots) and drop explicit scaling
- Keep thresholds/configurable constants; document in code
- Modify in place: src/imdb_recommender/recommender_elasticnet.py

Phase 5 — Results and Docs reduction
8) Results policy
- Keep only:
  - results/metrics/*.json (eval outputs)
  - results/tuning/*.{json,csv} (best params + CV tables from dev script)
- Archive verbose E2E outputs and long reports; keep ELASTICNET_CV_FINAL_REPORT.md as baseline evidence

9) Docs policy
- Keep: README.md, docs/DEVELOPMENT.md (short, actionable)
- Archive long-form analyses; link from DEVELOPMENT.md if needed

Phase 6 — Tests, Metric Guardrails, and CI
10) Test suite (protect current behavior and metrics)
- Unit tests:
  - Data IO: parsing/normalization; content-type filtering (movies vs tv) behavior
  - Feature engineering: ElasticNet _engineer_features yields stable shapes on fixtures
  - Ranker: normalization/blending determinism
  - Metrics: RMSE/MAE/precision@K/NDCG@K correctness on toy data
- Integration tests:
  - CLI recommend with --content-type movies|tv returns deterministic top-K on fixtures
  - CLI eval produces metrics JSON with expected keys
- Performance guardrails (non-flaky):
  - Fixed random_state, stable preprocessing; avoid leakage
  - Perf test (opt-in via env) using your real ratings dataset:
    - Reads E2E_RATINGS and E2E_WATCHLIST paths
    - Runs 5-fold CV for ElasticNet and asserts RMSE ≤ 1.60
      - Baseline from ELASTICNET_CV_FINAL_REPORT.md: 1.386 ± 0.095
  - Store baseline expectations in tests/baselines/metrics.json; compare with tolerances
- Coverage target: ≥ 85%
- Files (new):
  - tests/unit/test_data_io.py
  - tests/unit/test_filtering.py
  - tests/unit/test_features_elasticnet.py
  - tests/unit/test_ranker.py
  - tests/unit/test_metrics.py
  - tests/integration/test_cli_recommend.py
  - tests/integration/test_cli_eval.py
  - tests/perf/test_elasticnet_cv_guard.py  (skipped unless env vars present)

11) CI
- Lint + unit/integration on PRs
- Optional workflow_dispatch job for perf guard on provided dataset
- Upload small artifacts (results/metrics/*.json) for eval job

Acceptance Criteria
- One-line recommend works from two files and supports content filter:
  - imdbrec recommend --ratings-file data/ratings.csv --watchlist-file data/watchlist.csv --model svd --topk 10 --content-type tv
- Eval available and documented; supports optional CV summary:
  - imdbrec eval --ratings-file ... --watchlist-file ... --model elasticnet --cv 5 --out results/metrics/elasticnet_eval.json
  - Outputs concise table and JSON with RMSE/MAE and optional ranking@K
- No public CLI for tuning
  - Dev script scripts/diagnostics/tune.py produces results/tuning/{model}_best_params.json and {model}_cv_results.csv
- Consistent 0–1 score scale across models for blending/display
- ElasticNet pipeline completed with minimal changes; no dead code
- Results/docs trimmed
- Tests prevent regressions; ElasticNet RMSE guard ≤ 1.60 on full dataset (opt-in)

Reuse-first principles (guidance)
- Prefer moving code over rewriting; keep public signatures/import paths stable
- Encapsulate legacy behaviors behind thin adapters rather than removing them
- Use deprecation warnings for old CLI commands that now forward to recommend
- Document any intentional metric changes and update baselines with evidence

Timeline
- Week 1: Phase 1–2 (CLI with --content-type, Data IO, evaluation)
- Week 2: Phase 3–4 (Internal model-selection + dev tuning script; ElasticNet completion)
- Week 3: Phase 5–6 (Docs/Results trimming, tests, CI; add perf guard)

Risks and Mitigations
- Dataset variability may cause metric drift
  - Use stratified/time-based splits with fixed seeds; keep tolerance bands
- Backward-compat impacts for scripts
  - Archive old scripts with deprecation notes; keep aliases
- Flaky tests from randomness
  - Seed everything; avoid leakage; keep perf tests opt-in

Deliverable Impact
- Simpler UX: recommend + eval only, with TV/Movie filters
- Reproducible diagnostics: consistent metrics with concise artifacts
- Lower maintenance: fewer scripts, unified APIs, reuse current code
- Early detection of regressions via guardrail

## Phase 1 — Execution Metaprompt for AI Agent

You are an autonomous code agent working in this repository to complete Phase 1 (Core API and CLI simplification) of the uplift plan. Follow these instructions exactly. Make minimal, surgical changes that reuse existing code. After every substantive step, update the “Uplift Plan Progress Tracker” at the bottom of this document with a short note, a link to the commit hash, and a checkbox.

Operating rules
- Branching: create a feature branch named feat/phase1-cli-data. Commit early and often with conventional commit messages (e.g., feat(cli): add --content-type).
- Safety: do not remove code paths without a compatibility shim or deprecation note. Prefer forwarding/aliases.
- Scope: limit changes to Phase 1 files unless a tiny change is required elsewhere to keep things compiling.
- Tests: run the existing test suite (if present) and add a tiny smoke test only if needed to validate CLI changes.
- Lint/format: run ruff/black or the project’s configured linters/formatters if present.
- Progress logging: update the Progress Tracker section with each completed step.

System context
- Repo root: /Users/brent/workspace/imdb_recommender_pkg
- Python version: use the project’s pinned version (pyproject.toml or requirements.txt).
- CLI entry file: src/imdb_recommender/cli.py
- Data IO: src/imdb_recommender/data_io.py
- Ranker: src/imdb_recommender/ranker.py
- Base interface: src/imdb_recommender/recommender_base.py
- Models: src/imdb_recommender/recommender_svd.py, src/imdb_recommender/recommender_elasticnet.py

Objectives
1) Consolidate CLI to two public commands: recommend and eval.
2) Add --content-type movies|tv|all to recommend; keep legacy “top-watchlist-*” aliases as thin wrappers.
3) Centralize data normalization in data_io; expose a single load_dataset() path and a content filter helper.
4) Align score semantics by normalizing in ranker (0–1) without altering model internals in Phase 1.
5) Ensure no regressions: run smoke flows, update help texts, and document changes.

Detailed step plan

Step A — Create branch and baseline
- Create branch feat/phase1-cli-data.
- Ensure repo installs in editable mode:
  - python3 -m pip install -e .
- Run format/lint and tests (if configured):
  - ruff check . or flake8; black .; pytest -q
- Update Progress Tracker with baseline status.

Step B — Audit current CLI
- Open src/imdb_recommender/cli.py; list existing top-level commands and options.
- Identify commands to deprecate/forward (e.g., top-watchlist-movies, top-watchlist-tv).
- Decide whether CLI uses click or argparse; keep the existing library.
- Add a short summary of the current CLI surface to the Progress Tracker.

Step C — Implement recommend with --content-type
- In src/imdb_recommender/cli.py:
  - Ensure a recommend command exists (or create one if missing).
  - Add option --content-type with choices movies|tv|all (default=all).
  - Wire the option into the data loading path (see Step D) and final filtering.
  - Preserve existing options: --ratings-file, --watchlist-file, --model, --topk, etc.
  - Add legacy alias commands (top-watchlist-movies, top-watchlist-tv) that call recommend with --content-type pre-set and print a deprecation warning to stderr.
- Update CLI help texts and examples.

Step D — Centralize data normalization and filtering
- In src/imdb_recommender/data_io.py:
  - Add or reuse a function like load_dataset(ratings_path, watchlist_path, *, random_state: int | None = 42) -> Dataset that:
    - Reads both files and returns standardized columns: imdb_const, my_rating, title_type, year, imdb_rating, num_votes, genres, directors, release_date, rated_at.
    - Applies one consistent renaming/mapping path (move any rename_map from model code into here).
  - Add helper filter_by_content_type(df: pd.DataFrame, kind: Literal["movies","tv","all"]) -> pd.DataFrame
    - movies set: {"movie", "tvMovie"}
    - tv set: {"tvSeries", "tvMiniSeries", "tvSpecial"}
    - Exclude episodes by default (e.g., "tvEpisode") unless kind=="all".
  - Make this helper available to CLI; do not duplicate the logic elsewhere.
- Keep serialization types and Dataset dataclass/schema unchanged unless strictly necessary.

Step E — Normalize scores in ranker (Phase 1 only)
- In src/imdb_recommender/ranker.py:
  - If not present, add a small normalize_scores(scores: dict[str, float]) -> dict[str, float] that min-max scales to [0,1] when spread > 0; otherwise return 0.5 for all.
  - Ensure recommend code path uses normalized scores before ranking/blending.
- Do not change model outputs in Phase 1.

Step F — Wire the flow end-to-end
- In src/imdb_recommender/cli.py recommend handler:
  - Use load_dataset(...) to build Dataset.
  - Apply filter_by_content_type on the watchlist side (and/or candidate set), per the plan.
  - Instantiate the selected model via the existing factory/registry (reuse RecommenderAlgo).
  - Generate scores, normalize in ranker, and output top-K with titles, years, and scores.
- Ensure eval command continues to work or prints “Not yet implemented” if Phase 2 is pending; do not remove it.

Step G — Deprecation wrappers and messaging
- Keep legacy commands; forward to recommend with appropriate --content-type.
- Print a one-line deprecation notice suggesting the new command.
- Add a TODO comment with removal date or Phase number.

Step H — Smoke validation
- Run a local smoke check using any small sample ratings/watchlist in the repo:
  - imdbrec recommend --ratings-file <path> --watchlist-file <path> --model elasticnet --topk 5 --content-type movies
  - imdbrec recommend --ratings-file <path> --watchlist-file <path> --model elasticnet --topk 5 --content-type tv
- If no sample data exists, create tiny fixtures under tests/fixtures/ for local validation only in this branch.
- Verify the command exits 0, outputs sensible items, and does not crash.

Step I — Documentation touch-ups
- Update CLI usage examples in README.md (or add a short CLI section if missing).
- Add a brief note in docs/DEVELOPMENT.md on the new content-type switch and deprecation of legacy commands.

Step J — Finalize Phase 1 PR
- Ensure all changes are committed and linters/tests pass.
- Open a PR titled “Phase 1: CLI consolidation + data normalization (+ content-type)”.
- In the PR description, paste:
  - What changed (bullets)
  - Before/after CLI examples
  - Any known follow-ups deferred to Phase 2–4
- Update the Progress Tracker with the PR link and set Phase 1 status to “Complete (awaiting review)” or “Merged”.

Acceptance for Phase 1
- CLI exposes recommend and eval (eval can be a no-op stub until Phase 2).
- recommend supports --content-type movies|tv|all and works end-to-end with existing models.
- Legacy commands forward to recommend and print a deprecation note.
- Data normalization occurs in data_io; models do not duplicate column mapping.
- Scores are normalized to 0–1 for ranking in ranker.
- Smoke runs succeed; no crashes or obvious regressions.

Suggested commands (macOS terminal)
- Create branch: git checkout -b feat/phase1-cli-data
- Install editable: python3 -m pip install -e .
- Lint/format (if configured): ruff check . && black .
- Run CLI smoke: imdbrec --help; imdbrec recommend --help
- Run sample: imdbrec recommend --ratings-file data/ratings.csv --watchlist-file data/watchlist.csv --model elasticnet --topk 5 --content-type movies

## Uplift Plan Progress Tracker
- [ ] Phase 1 kickoff: branch feat/phase1-cli-data created
- [ ] Baseline checks: install, lint, tests (if present) pass
- [ ] CLI audited; legacy commands identified and documented
- [ ] recommend command updated with --content-type movies|tv|all
- [ ] Legacy commands forward to recommend with deprecation warnings
- [ ] Data normalization centralized in data_io; single load_dataset entry point
- [ ] filter_by_content_type helper implemented and reused
- [ ] Ranker normalizes scores to 0–1 before ranking/blending
- [ ] End-to-end smoke test for movies and tv completed
- [ ] README/DEVELOPMENT docs updated with new CLI usage
- [ ] PR opened for Phase 1 and linked here