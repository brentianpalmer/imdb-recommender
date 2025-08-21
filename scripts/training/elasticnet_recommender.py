#!/usr/bin/env python3
# Canonical location: scripts/training/elasticnet_recommender.py
# Expected inputs: ratings CSV and watchlist file

"""
ElasticNet Movie Recommender
Generates movie recommendations using ElasticNet with engineered features.
"""

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
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
                print(
                    f"   ⚠️  Found {content[col].isna().sum()} missing IMDb ratings, filling with {global_median_imdb:.1f}"  # noqa: E501
                )
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


def standardize_features(X_train, X_predict, numeric_cols):
    """Standardize numeric features using training set statistics."""
    X_train = X_train.copy()
    X_predict = X_predict.copy()

    mu = X_train[numeric_cols].mean()
    sd = X_train[numeric_cols].std().replace(0, 1.0)

    X_train.loc[:, numeric_cols] = (X_train[numeric_cols] - mu) / sd
    X_predict.loc[:, numeric_cols] = (X_predict[numeric_cols] - mu) / sd

    return X_train, X_predict


def load_watchlist_data(watchlist_path):
    """Load watchlist data from Excel or CSV."""
    if watchlist_path.endswith(".xlsx"):
        return pd.read_excel(watchlist_path)
    else:
        return pd.read_csv(watchlist_path)


def generate_elasticnet_recommendations(
    ratings_file, watchlist_file, topk=10, alpha=0.1, l1_ratio=0.1
):
    """Generate top-k recommendations using trained ElasticNet model."""

    print("🔬 Loading data for ElasticNet recommendations...")

    # Load training data (ratings)
    ratings_df = pd.read_csv(ratings_file)
    print(f"   📊 Loaded {len(ratings_df)} rated movies")

    # Load prediction data (watchlist)
    watchlist_df = load_watchlist_data(watchlist_file)
    print(f"   📋 Loaded {len(watchlist_df)} watchlist movies")

    # Filter out future movies (after 2025)
    current_year = 2025
    if "Year" in watchlist_df.columns:
        # Convert Year to numeric and filter
        watchlist_df["Year"] = pd.to_numeric(watchlist_df["Year"], errors="coerce")
        before_filter = len(watchlist_df)
        watchlist_df = watchlist_df[
            watchlist_df["Year"].isna() | (watchlist_df["Year"] <= current_year)
        ]
        after_filter = len(watchlist_df)
        if before_filter > after_filter:
            print(
                f"   🚫 Filtered out {before_filter - after_filter} future movies (after {current_year})"  # noqa: E501
            )
            print(f"   📋 Remaining watchlist: {after_filter} movies")

    # Engineer features for training data
    print("🔧 Engineering features...")
    X_train, y_train, numeric_cols = engineer_features(ratings_df, top_dir_k=30)
    training_imdb_median = ratings_df[
        "IMDb Rating"
    ].median()  # Use training median for watchlist nulls
    print(f"   🎯 Training features: {X_train.shape[1]} dimensions")
    print(f"   📊 Training IMDb median: {training_imdb_median:.1f}")

    # Engineer features for watchlist (prediction data)
    X_watchlist, _, _ = engineer_features(watchlist_df, top_dir_k=30)

    # Align feature columns between training and prediction data
    all_columns = set(X_train.columns) | set(X_watchlist.columns)
    for col in all_columns:
        if col not in X_train.columns:
            X_train[col] = 0.0
        if col not in X_watchlist.columns:
            X_watchlist[col] = 0.0

    # Ensure same column order
    X_train = X_train.reindex(columns=sorted(all_columns)).fillna(0.0)
    X_watchlist = X_watchlist.reindex(columns=sorted(all_columns)).fillna(0.0)

    # Update numeric_cols to match available columns
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    print(f"   🎯 Aligned features: {X_train.shape[1]} dimensions")

    # Standardize features
    X_train_scaled, X_watchlist_scaled = standardize_features(X_train, X_watchlist, numeric_cols)

    # Train ElasticNet model
    print(f"🤖 Training ElasticNet model (α={alpha}, l1_ratio={l1_ratio})...")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # Generate predictions
    print(f"🔮 Generating predictions for {len(watchlist_df)} movies...")
    raw_predictions = model.predict(X_watchlist_scaled)

    # Transform predictions to 1-10 range using sigmoid-like scaling
    # This is more natural than hard clipping
    def scale_predictions(preds):
        # Use a sigmoid-like transformation to map to 1-10 range
        # First normalize predictions relative to training target range
        _y_min, _y_max = y_train.min(), y_train.max()
        y_mean, y_std = y_train.mean(), y_train.std()

        # Z-score normalize predictions
        pred_normalized = (preds - y_mean) / y_std

        # Apply tanh to bound between -1 and 1, then scale to 1-10
        pred_bounded = np.tanh(pred_normalized / 2)  # Gentle sigmoid
        pred_scaled = 5.5 + 4.5 * pred_bounded  # Map to 1-10 range (centered at 5.5)

        return np.clip(pred_scaled, 1.0, 10.0)  # Final safety clip

    predictions = scale_predictions(raw_predictions)
    print(f"   📊 Raw prediction range: {raw_predictions.min():.2f} - {raw_predictions.max():.2f}")
    print(f"   📊 Scaled prediction range: {predictions.min():.2f} - {predictions.max():.2f}")

    # Create recommendations dataframe
    recommendations = watchlist_df.copy()
    recommendations["predicted_rating"] = predictions
    recommendations = recommendations.sort_values("predicted_rating", ascending=False)

    # Filter out any movies that might be already rated (if tconst columns exist)
    if "Const" in ratings_df.columns and "Const" in recommendations.columns:
        rated_movies = set(ratings_df["Const"])
        recommendations = recommendations[~recommendations["Const"].isin(rated_movies)]
    elif "tconst" in ratings_df.columns and "tconst" in recommendations.columns:
        rated_movies = set(ratings_df["tconst"])
        recommendations = recommendations[~recommendations["tconst"].isin(rated_movies)]

    return recommendations.head(topk)


def main():
    parser = argparse.ArgumentParser(description="ElasticNet Movie Recommender")
    parser.add_argument("--ratings_file", type=str, required=True, help="Path to ratings CSV file")
    parser.add_argument(
        "--watchlist_file", type=str, required=True, help="Path to watchlist file (CSV or Excel)"
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--alpha", type=float, default=0.1, help="ElasticNet alpha parameter")
    parser.add_argument("--l1_ratio", type=float, default=0.1, help="ElasticNet l1_ratio parameter")
    parser.add_argument("--export_csv", type=str, default="", help="Export recommendations to CSV")

    args = parser.parse_args()

    print("🎬 ElasticNet Movie Recommender")
    print("=" * 50)

    try:
        recommendations = generate_elasticnet_recommendations(
            ratings_file=args.ratings_file,
            watchlist_file=args.watchlist_file,
            topk=args.topk,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
        )

        print(f"\n🏆 Top {args.topk} ElasticNet Recommendations:")
        print("=" * 70)

        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            title = row.get("Title", row.get("title", "Unknown Title"))
            year = row.get("Year", "")
            genres = row.get("Genres", row.get("genres", ""))
            pred_rating = row["predicted_rating"]

            print(f"{i:2}. {title} ({year})")
            print(f"    🎯 Predicted Rating: {pred_rating:.2f}  🎬 {genres}")
            print("    💡 ElasticNet feature-engineered prediction")
            print()

        if args.export_csv:
            recommendations.to_csv(args.export_csv, index=False)
            print(f"💾 Exported recommendations to: {args.export_csv}")

    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
