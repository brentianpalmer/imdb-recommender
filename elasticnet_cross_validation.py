#!/usr/bin/env python3
"""
ElasticNet Cross Validation
Runs comprehensive cross validation for the ElasticNet movie recommender.
"""

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_STATE = 42


def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _split_listfield(s):
    if pd.isna(s):
        return []
    return [t.strip() for t in str(s).split(",") if t and str(t).strip()]


def engineer_features(df, top_dir_k=30):
    """Build the feature matrix X and target y from the ratings export."""
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

    # Target (only for training data)
    if "your_rating" in df.columns:
        y = _safe_numeric(df["your_rating"]).astype(float).values
    else:
        y = None

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

    # Numerics - handle missing values properly
    for col in ["Year", "runtime", "imdb", "votes"]:
        content[col] = _safe_numeric(content[col])
        if col == "imdb":
            # Handle missing IMDb ratings by using global median from training data
            if content[col].isna().any():
                global_median_imdb = content[col].median() if content[col].notna().any() else 7.0
                content[col] = content[col].fillna(global_median_imdb)
        else:
            content[col] = content[col].fillna(content[col].median())

    # Cap extreme values to prevent feature explosion
    content["votes"] = content["votes"].clip(upper=2_000_000)  # Cap at 2M votes
    content["log_votes"] = np.log1p(content["votes"])
    content["Year"] = content["Year"].clip(lower=1900, upper=2025)  # Reasonable year range
    content["decade"] = (content["Year"] // 10) * 10

    # Dates - handle missing date_rated for watchlist
    if "date_rated" in df.columns and df["date_rated"].notna().any():
        content["date_rated_dt"] = pd.to_datetime(content["date_rated"], errors="coerce", utc=False)
        first_rate = content["date_rated_dt"].min()
        if pd.isna(first_rate):
            content["days_since_first_rate"] = 0.0
            content["rate_year"] = 0
            content["rate_month"] = 0
            content["rate_dow"] = 0
        else:
            content["days_since_first_rate"] = (
                content["date_rated_dt"] - first_rate
            ).dt.days.astype("float64")
            content["rate_year"] = (
                content["date_rated_dt"].dt.year.astype("Int64").fillna(0).astype(int)
            )
            content["rate_month"] = (
                content["date_rated_dt"].dt.month.astype("Int64").fillna(0).astype(int)
            )
            content["rate_dow"] = (
                content["date_rated_dt"].dt.dayofweek.astype("Int64").fillna(0).astype(int)
            )
    else:
        # For watchlist items without rating dates
        content["days_since_first_rate"] = 0.0
        content["rate_year"] = 0
        content["rate_month"] = 0
        content["rate_dow"] = 0

    content["release_dt"] = pd.to_datetime(content["release_date"], errors="coerce", utc=False)
    if content["release_dt"].notna().any():
        content["rel_month"] = content["release_dt"].dt.month.astype("Int64").fillna(0).astype(int)
        # Use a fixed reference date for age calculation for watchlist items
        ref_date = pd.Timestamp.now()
        age = (ref_date - content["release_dt"]).dt.days.astype("float64")
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

    return X, y, numeric_cols


def standardize_within_fold(X_train, X_test, numeric_cols):
    """Standardize numeric features using training set statistics."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    mu = X_train[numeric_cols].mean()
    sd = X_train[numeric_cols].std().replace(0, 1.0)

    X_train.loc[:, numeric_cols] = (X_train[numeric_cols] - mu) / sd
    X_test.loc[:, numeric_cols] = (X_test[numeric_cols] - mu) / sd

    return X_train, X_test


def clip_predictions(y_pred, min_val=1.0, max_val=10.0):
    """Simple clipping to bound predictions to [min_val, max_val] range."""
    return np.clip(y_pred, min_val, max_val)


def stratify_bins(y):
    """Bin continuous ratings into 5 bins for stratified splits."""
    s = pd.Series(y).astype(float)
    labels = [1, 3, 5, 7, 9]
    return pd.cut(s, bins=[0, 2, 4, 6, 8, 10], labels=labels, include_lowest=True).astype(int)


def run_cross_validation(
    ratings_file,
    n_splits=5,
    alphas=None,
    l1_ratios=None,
    top_dir_k=30,
    output_file="elasticnet_cv_results.csv",
):
    """Run comprehensive cross validation for ElasticNet model."""

    if alphas is None:
        alphas = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    print("🎯 Starting ElasticNet Cross Validation")
    print(f"   📊 Data: {ratings_file}")
    print(f"   🔢 CV Folds: {n_splits}")
    print(f"   📈 Alpha values: {alphas}")
    print(f"   📉 L1 ratios: {l1_ratios}")
    print(f"   🎬 Top directors: {top_dir_k}")
    print(f"   💾 Output: {output_file}")
    print()

    # Load data
    print("📥 Loading ratings data...")
    df = pd.read_csv(ratings_file)
    print(f"   Loaded {len(df):,} ratings")

    # Filter out future movies (after 2025)
    if "Year" in df.columns:
        future_mask = pd.to_numeric(df["Year"], errors="coerce") > 2025
        if future_mask.any():
            n_future = future_mask.sum()
            print(f"   🚫 Filtering out {n_future} future movies (after 2025)")
            df = df[~future_mask].copy()
            print(f"   📋 Remaining: {len(df):,} ratings")

    # Engineer features
    print("🔧 Engineering features...")
    X, y, numeric_cols = engineer_features(df, top_dir_k=top_dir_k)
    print(f"   Generated {X.shape[1]} features from {len(df)} movies")

    # Remove any rows with missing targets
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"   Valid samples: {len(y):,}")

    # Setup stratified K-fold
    y_bins = stratify_bins(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Results storage
    results = []

    # Grid search over hyperparameters
    total_combinations = len(alphas) * len(l1_ratios)
    combination = 0

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            combination += 1
            print(
                f"\n🧪 Testing combination {combination}/{total_combinations}: alpha={alpha}, l1_ratio={l1_ratio}"
            )

            fold_scores = []
            fold_r2s = []

            # Cross validation folds
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
                print(f"   Fold {fold + 1}/{n_splits}...", end="")

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Standardize features within fold
                X_train_std, X_val_std = standardize_within_fold(X_train, X_val, numeric_cols)

                # Train model
                model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=2000
                )
                model.fit(X_train_std, y_train)

                # Predict and clip to valid range
                y_pred_raw = model.predict(X_val_std)
                y_pred = clip_predictions(y_pred_raw, min_val=1.0, max_val=10.0)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)

                fold_scores.append(rmse)
                fold_r2s.append(r2)

                print(f" RMSE={rmse:.3f}, R²={r2:.3f}")

            # Store results
            mean_rmse = np.mean(fold_scores)
            std_rmse = np.std(fold_scores)
            mean_r2 = np.mean(fold_r2s)
            std_r2 = np.std(fold_r2s)

            results.append(
                {
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                    "mean_r2": mean_r2,
                    "std_r2": std_r2,
                    "fold_rmses": fold_scores,
                    "fold_r2s": fold_r2s,
                }
            )

            print(
                f"   📊 Average: RMSE={mean_rmse:.3f} ± {std_rmse:.3f}, R²={mean_r2:.3f} ± {std_r2:.3f}"
            )

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_rmse")

    print("\n🎉 Cross validation completed!")
    print(f"📊 Results saved to: {output_file}")

    # Show top 5 results
    print("\n🏆 Top 5 hyperparameter combinations:")
    print(
        results_df.head()[
            ["alpha", "l1_ratio", "mean_rmse", "std_rmse", "mean_r2", "std_r2"]
        ].to_string(index=False)
    )

    # Save results
    results_df.to_csv(output_file, index=False)

    # Best model summary
    best = results_df.iloc[0]
    print("\n🥇 Best model:")
    print(f"   Alpha: {best['alpha']}")
    print(f"   L1 ratio: {best['l1_ratio']}")
    print(f"   RMSE: {best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}")
    print(f"   R²: {best['mean_r2']:.4f} ± {best['std_r2']:.4f}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run ElasticNet Cross Validation")
    parser.add_argument(
        "--ratings_file", type=str, default="data/raw/ratings.csv", help="Path to ratings CSV file"
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--alphas", type=str, default="0.01,0.1,0.3,1,3,10", help="Comma-separated alpha values"
    )
    parser.add_argument(
        "--l1_ratios", type=str, default="0.1,0.5,0.9", help="Comma-separated L1 ratio values"
    )
    parser.add_argument(
        "--top_dir_k", type=int, default=30, help="Number of top directors to include"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/elasticnet_cv_results.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()

    # Parse hyperparameters
    alphas = [float(x.strip()) for x in args.alphas.split(",")]
    l1_ratios = [float(x.strip()) for x in args.l1_ratios.split(",")]

    # Create results directory
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Run cross validation
    run_cross_validation(
        ratings_file=args.ratings_file,
        n_splits=args.n_splits,
        alphas=alphas,
        l1_ratios=l1_ratios,
        top_dir_k=args.top_dir_k,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
