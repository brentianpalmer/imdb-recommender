#!/usr/bin/env python3
"""
run_elasticnet_cv.py
Replicates Elastic Net CV result on IMDb ratings export with engineered features.

Usage:
  python run_elasticnet_cv.py --ratings_file /path/to/ratings.csv \
      --n_splits 5 --alphas 0.01,0.1,0.3,1,3,10 --l1_ratios 0.1,0.5,0.9 \
      --top_dir_k 30 --out_csv results_elasticnet_cv.csv
"""
import argparse
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_STATE = 42

# ----------------------------- Utilities -----------------------------


def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _split_listfield(s):
    if pd.isna(s):
        return []
    return [t.strip() for t in str(s).split(",") if t and str(t).strip()]


def stratify_bins(y):
    """Bin continuous ratings into 5 bins for stratified splits."""
    s = pd.Series(y).astype(float)
    labels = [1, 3, 5, 7, 9]
    return pd.cut(s, bins=[0, 2, 4, 6, 8, 10], labels=labels, include_lowest=True).astype(int)


def standardize_within_fold(X_tr, X_te, numeric_cols):
    """Z-score numeric columns using train fold stats; leave one-hots as is."""
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    mu = X_tr[numeric_cols].mean()
    sd = X_tr[numeric_cols].std().replace(0, 1.0)
    X_tr.loc[:, numeric_cols] = (X_tr[numeric_cols] - mu) / sd
    X_te.loc[:, numeric_cols] = (X_te[numeric_cols] - mu) / sd
    return X_tr, X_te


# ------------------------- Feature Engineering ------------------------


def engineer_features(df, top_dir_k=30):
    """
    Build the feature matrix X and target y from the ratings export.
    Expected columns (IMDb CSV):
      Const, Your Rating, Genres, Title Type, Year, Runtime (mins),
      IMDb Rating, Num Votes, Date Rated, Release Date, Directors
    """
    # Rename to consistent names if present
    rename_map = {
        "Const": "tconst",
        "Your Rating": "your_rating",
        "Title Type": "titleType",
        "Runtime (mins)": "runtime",
        "IMDb Rating": "imdb",
        "Num Votes": "votes",
        "Date Rated": "date_rated",
        "Release Date": "release_date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Target
    if "your_rating" not in df.columns:
        raise ValueError("Expected 'Your Rating' column in ratings CSV.")
    y = _safe_numeric(df["your_rating"]).astype(float).values

    # Base frame for features
    cols_needed = [
        "tconst",
        "Genres",
        "titleType",
        "Year",
        "runtime",
        "imdb",
        "votes",
        "date_rated",
        "release_date",
        "Directors",
    ]
    content = pd.DataFrame()
    for c in cols_needed:
        content[c] = df[c] if c in df.columns else np.nan

    # Numerics
    for col in ["Year", "runtime", "imdb", "votes"]:
        content[col] = _safe_numeric(content[col])
        content[col] = content[col].fillna(content[col].median())
    content["log_votes"] = np.log1p(content["votes"])
    content["decade"] = (content["Year"] // 10) * 10

    # Dates
    content["date_rated_dt"] = pd.to_datetime(content["date_rated"], errors="coerce", utc=False)
    content["release_dt"] = pd.to_datetime(content["release_date"], errors="coerce", utc=False)

    first_rate = content["date_rated_dt"].min()
    if pd.isna(first_rate):
        # If absent, fall back to zeros
        content["days_since_first_rate"] = 0.0
        content["rate_year"] = 0
        content["rate_month"] = 0
        content["rate_dow"] = 0
    else:
        content["days_since_first_rate"] = (content["date_rated_dt"] - first_rate).dt.days.astype(
            "float64"
        )
        content["rate_year"] = (
            content["date_rated_dt"].dt.year.astype("Int64").fillna(0).astype(int)
        )
        content["rate_month"] = (
            content["date_rated_dt"].dt.month.astype("Int64").fillna(0).astype(int)
        )
        content["rate_dow"] = (
            content["date_rated_dt"].dt.dayofweek.astype("Int64").fillna(0).astype(int)
        )

    if content["release_dt"].notna().any():
        content["rel_month"] = content["release_dt"].dt.month.astype("Int64").fillna(0).astype(int)
        age = (content["date_rated_dt"] - content["release_dt"]).dt.days.astype("float64")
        content["age_at_rating_days"] = age.fillna(age.median())
    else:
        content["rel_month"] = 0
        content["age_at_rating_days"] = 0.0

    # One-hot categorical
    onehot_title = pd.get_dummies(content["titleType"].fillna("Unknown"), prefix="type")
    onehot_decade = pd.get_dummies(content["decade"].fillna(0).astype(int), prefix="dec")
    onehot_rate_month = pd.get_dummies(content["rate_month"].fillna(0).astype(int), prefix="rate_m")
    onehot_rate_dow = pd.get_dummies(content["rate_dow"].fillna(0).astype(int), prefix="rate_dow")
    onehot_rel_month = pd.get_dummies(content["rel_month"].fillna(0).astype(int), prefix="rel_m")

    # Genres (multi-hot)
    genres_tokens = (
        content["Genres"]
        .fillna("")
        .apply(lambda s: [g.strip() for g in str(s).split(",") if g.strip()])
    )
    mlb = MultiLabelBinarizer()
    genres_mh = pd.DataFrame(
        mlb.fit_transform(genres_tokens),
        columns=[f"g_{g}" for g in mlb.classes_],
        index=content.index,
    )

    # Directors → Top-K one-hot
    directors_lists = content["Directors"].apply(lambda s: _split_listfield(s))
    dir_counts = Counter([d for lst in directors_lists for d in lst])
    top_dirs = set([d for d, _ in dir_counts.most_common(top_dir_k)])

    def _norm_name(d):
        return re.sub(r"[^A-Za-z0-9]+", "_", d)[:30]

    dir_cols = [f"dir_{_norm_name(d)}" for d in top_dirs]
    dir_features = pd.DataFrame(0, index=content.index, columns=dir_cols, dtype=float)
    name_map = {d: f"dir_{_norm_name(d)}" for d in top_dirs}
    for idx, lst in directors_lists.items():
        for d in lst:
            if d in name_map:
                dir_features.at[idx, name_map[d]] = 1.0
    content["has_director_info"] = (directors_lists.str.len() > 0).astype(int)

    # Numeric columns
    numeric_cols = [
        "Year",
        "runtime",
        "imdb",
        "log_votes",
        "days_since_first_rate",
        "age_at_rating_days",
    ]

    # Assemble X
    X = pd.concat(
        [
            onehot_title,
            onehot_decade,
            onehot_rate_month,
            onehot_rate_dow,
            onehot_rel_month,
            genres_mh,
            dir_features,
            content[numeric_cols].astype(float),
        ],
        axis=1,
    ).fillna(0.0)

    meta = {
        "n_rows": len(df),
        "n_features": X.shape[1],
        "numeric_cols": numeric_cols,
        "top_dir_count": len(dir_cols),
    }
    return X, y, meta


def elasticnet_cv(X, y, bins, alphas, l1_ratios, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    numeric_cols = [
        c
        for c in X.columns
        if c
        in ["Year", "runtime", "imdb", "log_votes", "days_since_first_rate", "age_at_rating_days"]
    ]

    rows = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            fold_rmses = []
            for tr_idx, te_idx in skf.split(X, bins):
                X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
                y_tr, y_te = y[tr_idx], y[te_idx]

                # Scale numeric columns (per fold)
                X_tr, X_te = standardize_within_fold(X_tr, X_te, numeric_cols)

                model = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                rmse = math.sqrt(mean_squared_error(y_te, preds))
                fold_rmses.append(rmse)

            rows.append(
                {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "rmse_mean": float(np.mean(fold_rmses)),
                    "rmse_std": float(np.std(fold_rmses)),
                    "n_splits": n_splits,
                }
            )

    res = (
        pd.DataFrame(rows)
        .sort_values(["rmse_mean", "rmse_std", "alpha", "l1_ratio"])
        .reset_index(drop=True)
    )
    return res


def main():

    parser = argparse.ArgumentParser(
        description="Elastic Net CV on IMDb ratings with engineered features."
    )
    parser.add_argument(
        "--ratings_file", type=str, required=True, help="Path to IMDb ratings CSV export"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.1,0.3,1,3,10",
        help="Comma-separated list of alpha values (e.g., '0.01,0.1,0.3,1,3,10')",
    )
    parser.add_argument(
        "--l1_ratios",
        type=str,
        default="0.1,0.5,0.9",
        help="Comma-separated list of l1_ratio values (e.g., '0.1,0.5,0.9')",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument(
        "--top_dir_k", type=int, default=30, help="Top-K directors to one-hot (default: 30)"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/elasticnet_cv_results.csv",
        help="Optional path to write CV results CSV",
    )
    args = parser.parse_args()

    # Load ratings
    df = pd.read_csv(args.ratings_file)

    # Build features
    X, y, meta = engineer_features(df, top_dir_k=args.top_dir_k)
    print(
        f"[INFO] Rows: {meta['n_rows']}, Features: {meta['n_features']} "
        f"(Top-{meta['top_dir_count']} directors)"
    )

    # Stratification labels
    bins = stratify_bins(y)

    # Parse hyperparameters
    alphas = [float(x) for x in args.alphas.split(",") if str(x).strip()]
    l1_ratios = [float(x) for x in args.l1_ratios.split(",") if str(x).strip()]
    print(
        f"[INFO] Grid: {len(alphas)} alphas × {len(l1_ratios)} l1_ratios = "
        f"{len(alphas)*len(l1_ratios)} configs"
    )

    # Run CV
    results = elasticnet_cv(X, y, bins, alphas, l1_ratios, n_splits=args.n_splits)
    print("\n=== Elastic Net CV (sorted by RMSE) ===")
    print(results.head(15).to_string(index=False))

    # Best row
    best = results.iloc[0].to_dict()
    print("\n=== Best Configuration ===")
    print(
        f"alpha={best['alpha']}, l1_ratio={best['l1_ratio']}, "
        f"RMSE_mean={best['rmse_mean']:.4f}, RMSE_std={best['rmse_std']:.4f}, "
        f"folds={int(best['n_splits'])}"
    )

    # Save CSV
    if args.out_csv:
        results.to_csv(args.out_csv, index=False)
        print(f"[INFO] Wrote CV results to: {args.out_csv}")


if __name__ == "__main__":
    main()
