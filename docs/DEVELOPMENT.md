## Development Notes

This document captures actionable guidance for working on the CLI and data
pipeline during the uplift plan.

### Public CLI surface
- `imdbrec recommend` is the single entry-point for generating recommendations.
  - Key options: `--ratings-file`, `--watchlist-file`, `--config`, `--model {svd,elasticnet}`, `--topk`, `--content-type {movies,tv,all}`.
  - Content filtering is implemented once and reused across the app:
    - movies → {"movie", "tvMovie"}
    - tv → {"tvSeries", "tvMiniSeries", "tvSpecial"}
- Legacy wrappers:
  - `imdbrec top-watchlist-movies` and `imdbrec top-watchlist-tv` are deprecated and forward to `recommend` with a warning.

### Data normalization
- Central ingress lives in `imdb_recommender.data_io.ingest_sources`, which normalizes
  both ratings and watchlist files and materializes normalized artefacts under `data/`.

### Ranking & score semantics
- Scores are normalized to 0–1 prior to ranking/blending inside `Ranker` to keep
  cross-model semantics consistent.

### Examples
```bash
# With explicit file paths
imdbrec recommend \
  --ratings-file data/raw/ratings.csv \
  --watchlist-file data/raw/watchlist.xlsx \
  --model svd \
  --topk 10 \
  --content-type all

# With config.toml
imdbrec recommend --config config.toml --model elasticnet --content-type movies --topk 15
```

### Notes
- The `eval` command is planned in Phase 2; it is not yet exposed publicly.
- Avoid duplicating column mapping logic in model code; prefer `data_io`.

