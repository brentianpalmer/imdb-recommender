#!/usr/bin/env python3
"""
Direct ratings comparison: actual ratings vs SVD predictions
Uses the same direct approach as export_watchlist.py
"""

import logging

import numpy as np
import pandas as pd

from imdb_recommender.data_io import Dataset
from imdb_recommender.recommender_svd import SVDAutoRecommender

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Export actual ratings vs SVD predictions for comparison analysis."""

    print("📊 EXPORTING RATINGS VS SVD PREDICTIONS COMPARISON")
    print("=" * 60)

    try:
        # Load the data directly (same as export_watchlist.py)
        ratings_df = pd.read_parquet("data/ratings_normalized.parquet")
        watchlist_df = pd.read_parquet("data/watchlist_normalized.parquet")

        print(f"✅ Loaded: {len(ratings_df)} ratings, {len(watchlist_df)} watchlist items")

        # Create dataset
        dataset = Dataset(ratings=ratings_df, watchlist=watchlist_df)

        # Create SVD with optimal hyperparameters (same as export_watchlist.py)
        svd = SVDAutoRecommender(dataset, random_seed=42)
        print(f"🎯 Using optimal SVD: {svd.hyperparams}")

        # Generate predictions for all rated items
        print(f"📋 Generating predictions for {len(ratings_df)} rated items")

        # Get SVD predictions for all items (including rated ones)
        svd_scores, svd_explanations = svd.score(
            seeds=[],
            user_weight=1.0,
            global_weight=0.0,
            recency=0.0,
            exclude_rated=False,  # Include rated items for comparison
        )
        print(f"🎯 Generated {len(svd_scores)} SVD predictions")

        output_data = []
        predictions_count = 0

        for _, row in ratings_df.iterrows():
            try:
                # Get SVD prediction using the same method as export_watchlist.py
                svd_score = svd_scores.get(row["imdb_const"], 0)

                if svd_score > 0:
                    svd_predicted_rating = 1 + 9 * svd_score  # Same scaling as export_watchlist.py
                    svd_predicted_rating = round(svd_predicted_rating, 2)
                    predictions_count += 1
                else:
                    svd_predicted_rating = None

                # Build output row
                output_row = {
                    "imdb_const": row["imdb_const"],
                    "title": row["title"],
                    "year": row["year"],
                    "genres": row["genres"],
                    "title_type": row["title_type"],
                    "imdb_rating": row["imdb_rating"],
                    "user_rating": row["my_rating"],  # Fixed column name
                    "svd_score_raw": svd_score,
                    "svd_predicted_rating": svd_predicted_rating,
                    "num_votes": row.get("num_votes", None),
                    "runtime_minutes": row.get("runtime_minutes", None),
                }

                # Add comparison metrics
                if svd_predicted_rating is not None:
                    output_row["rating_difference"] = svd_predicted_rating - row["my_rating"]
                    output_row["abs_difference"] = abs(output_row["rating_difference"])

                    # Categorize prediction quality
                    diff = abs(output_row["rating_difference"])
                    if diff <= 0.5:
                        output_row["prediction_quality"] = "excellent_match"
                    elif diff <= 1.0:
                        output_row["prediction_quality"] = "good_match"
                    elif diff <= 1.5:
                        output_row["prediction_quality"] = "fair_match"
                    else:
                        output_row["prediction_quality"] = "poor_match"
                else:
                    output_row["rating_difference"] = None
                    output_row["abs_difference"] = None
                    output_row["prediction_quality"] = "no_prediction"

                output_data.append(output_row)

            except Exception as e:
                logger.warning(f"Could not process {row['imdb_const']}: {e}")
                # Add row with no prediction
                output_row = {
                    "imdb_const": row["imdb_const"],
                    "title": row["title"],
                    "year": row["year"],
                    "genres": row["genres"],
                    "title_type": row["title_type"],
                    "imdb_rating": row["imdb_rating"],
                    "user_rating": row["my_rating"],  # Fixed column name
                    "svd_score_raw": None,
                    "svd_predicted_rating": None,
                    "rating_difference": None,
                    "abs_difference": None,
                    "prediction_quality": "no_prediction",
                    "num_votes": row.get("num_votes", None),
                    "runtime_minutes": row.get("runtime_minutes", None),
                }
                output_data.append(output_row)

        # Create DataFrame
        df = pd.DataFrame(output_data)
        print(f"🎯 Generated {predictions_count} SVD predictions")

        # Sort by absolute difference (best predictions first)
        df = df.sort_values("abs_difference", na_position="last")

        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))

        # Export to CSV
        output_file = "ratings_vs_svd_predictions.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Exported to: {output_file}")
        print(f"📊 Total items: {len(df)}")

        # Calculate accuracy metrics
        valid_df = df.dropna(subset=["svd_predicted_rating"])
        if len(valid_df) > 0:
            actual_ratings = valid_df["user_rating"]
            predicted_ratings = valid_df["svd_predicted_rating"]

            # Calculate metrics
            mae = np.mean(np.abs(actual_ratings - predicted_ratings))
            rmse = np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2))
            correlation = np.corrcoef(actual_ratings, predicted_ratings)[0, 1]

            print("\n📈 PREDICTION ACCURACY METRICS:")
            print(f"   Items with predictions: {len(valid_df)}")
            print(f"   Mean Absolute Error (MAE): {mae:.3f}")
            print(f"   Root Mean Square Error (RMSE): {rmse:.3f}")
            print(f"   Correlation coefficient: {correlation:.3f}")

            print("\n📊 RATING COMPARISON:")
            print(f"   Average your rating: {valid_df['user_rating'].mean():.2f}")
            print(f"   Average SVD prediction: {valid_df['svd_predicted_rating'].mean():.2f}")
            print(f"   Average absolute difference: {valid_df['abs_difference'].mean():.2f}")

            # Quality breakdown
            quality_counts = valid_df["prediction_quality"].value_counts()
            print("\n🎯 PREDICTION QUALITY BREAKDOWN:")
            for quality, count in quality_counts.items():
                pct = (count / len(valid_df)) * 100
                print(f"   {quality.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

            # Show best predictions
            best_predictions = valid_df.head(10)
            print("\n🏆 TOP 10 MOST ACCURATE PREDICTIONS:")
            for _, row in best_predictions.iterrows():
                print(f"    {row['rank']}. {row['title']} ({row['year']})")
                print(
                    f"       👤 Your: {row['user_rating']:.1f}  "
                    f"🎯 SVD: {row['svd_predicted_rating']:.2f}  "
                    f"📊 Diff: {row['rating_difference']:+.2f}"
                )

            # Show worst predictions
            worst_predictions = valid_df.tail(10)
            print("\n❌ TOP 10 LEAST ACCURATE PREDICTIONS:")
            for _, row in worst_predictions.iterrows():
                print(f"    {row['rank']}. {row['title']} ({row['year']})")
                print(
                    f"       👤 Your: {row['user_rating']:.1f}  "
                    f"🎯 SVD: {row['svd_predicted_rating']:.2f}  "
                    f"📊 Diff: {row['rating_difference']:+.2f}"
                )

        return output_file

    except Exception as e:
        logger.error(f"❌ Error during export: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
