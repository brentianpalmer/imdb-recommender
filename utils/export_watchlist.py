"""
Export Watchlist with IMDb Stats and SVD Predicted Ratings
"""

import os
import sys

import pandas as pd

# Add the package to path so we can import
sys.path.append(os.path.abspath("."))

from imdb_recommender.data_io import Dataset
from imdb_recommender.recommender_svd import SVDAutoRecommender


def export_watchlist_with_predictions():
    """Export comprehensive watchlist with IMDb stats and SVD predictions."""
    print("📊 EXPORTING WATCHLIST WITH SVD PREDICTIONS")
    print("=" * 60)

    try:
        # Load the data
        ratings_df = pd.read_parquet("data/ratings_normalized.parquet")
        watchlist_df = pd.read_parquet("data/watchlist_normalized.parquet")

        print(f"✅ Loaded: {len(ratings_df)} ratings, {len(watchlist_df)} watchlist items")

        # Create dataset
        dataset = Dataset(ratings=ratings_df, watchlist=watchlist_df)

        # Create SVD with optimal hyperparameters
        svd = SVDAutoRecommender(dataset, random_seed=42)
        print(f"🎯 Using optimal SVD: {svd.hyperparams}")

        # Get all watchlist items (excluding already rated)
        rated_ids = set(ratings_df["imdb_const"])
        watchlist_items = watchlist_df[~watchlist_df["imdb_const"].isin(rated_ids)].copy()

        print(f"📋 Watchlist items to predict: {len(watchlist_items)}")

        # Get SVD predictions for all watchlist items
        svd_scores, svd_explanations = svd.score(
            seeds=[], user_weight=0.7, global_weight=0.3, recency=0.0, exclude_rated=True
        )

        print(f"🎯 Generated {len(svd_scores)} SVD predictions")

        # Create comprehensive output dataframe
        output_data = []

        for _, item in watchlist_items.iterrows():
            imdb_id = item["imdb_const"]

            # Get SVD prediction (convert from 0-1 scale to 1-10 scale)
            svd_score = svd_scores.get(imdb_id, 0)
            predicted_rating = 1 + 9 * svd_score if svd_score > 0 else None

            # Get explanation
            explanation = svd_explanations.get(imdb_id, "")

            # Compile all data
            row = {
                "imdb_const": imdb_id,
                "title": item.get("title", ""),
                "year": item.get("year", ""),
                "genres": item.get("genres", ""),
                "title_type": item.get("title_type", ""),
                "imdb_rating": item.get("imdb_rating", ""),
                "num_votes": item.get("num_votes", ""),
                "runtime_minutes": item.get("runtime_minutes", ""),
                "directors": item.get("directors", ""),
                "writers": item.get("writers", ""),
                "actors": item.get("actors", ""),
                "plot": item.get("plot", ""),
                "svd_score_raw": svd_score,
                "svd_predicted_rating": predicted_rating,
                "svd_explanation": explanation,
            }

            output_data.append(row)

        # Create DataFrame and sort by predicted rating (descending)
        df = pd.DataFrame(output_data)
        df = df.sort_values("svd_predicted_rating", ascending=False, na_position="last")

        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))

        # Format numeric columns
        df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce").round(1)
        df["num_votes"] = pd.to_numeric(df["num_votes"], errors="coerce").astype("Int64")
        df["runtime_minutes"] = pd.to_numeric(df["runtime_minutes"], errors="coerce").astype(
            "Int64"
        )
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["svd_predicted_rating"] = df["svd_predicted_rating"].round(2)

        # Export to CSV
        output_file = "watchlist_with_svd_predictions.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"✅ Exported to: {output_file}")
        print(f"📊 Total items: {len(df)}")

        # Show summary statistics
        valid_predictions = df["svd_predicted_rating"].dropna()
        if len(valid_predictions) > 0:
            print("\n📈 SVD PREDICTION SUMMARY:")
            print(f"   Items with predictions: {len(valid_predictions)}")
            print(f"   Average predicted rating: {valid_predictions.mean():.2f}")
            print(f"   Highest predicted rating: {valid_predictions.max():.2f}")
            print(f"   Lowest predicted rating: {valid_predictions.min():.2f}")

        # Show top 10 predictions
        print("\n🏆 TOP 10 PREDICTED RATINGS:")
        top_10 = df.head(10)
        for _, row in top_10.iterrows():
            title = row["title"][:50] + "..." if len(str(row["title"])) > 50 else row["title"]
            year = f"({row['year']})" if pd.notna(row["year"]) else ""
            rating = row["svd_predicted_rating"]
            imdb = row["imdb_rating"]
            print(f"   {row['rank']:2d}. {title} {year}")
            print(f"       🎯 Predicted: {rating:.2f}  📊 IMDb: {imdb}  🎬 {row['title_type']}")

        # Show content type breakdown
        print("\n📋 CONTENT TYPE BREAKDOWN:")
        type_counts = df["title_type"].value_counts()
        for content_type, count in type_counts.items():
            avg_pred = df[df["title_type"] == content_type]["svd_predicted_rating"].mean()
            print(f"   {content_type}: {count} items (avg predicted: {avg_pred:.2f})")

        return output_file

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    export_watchlist_with_predictions()
