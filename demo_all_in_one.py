#!/usr/bin/env python3
"""
All-in-One Four-Stage IMDb Recommender Demo

This script demonstrates the full capabilities of the AllInOneRecommender:
- Four-stage recommendation process
- Intelligent explanations
- Model persistence
- Evaluation metrics
- CSV export with detailed scores

Run with: python demo_all_in_one.py

Author: IMDb Recommender Team
Date: August 2025
"""

import sys
from pathlib import Path

# Add the package to the path for demo purposes
sys.path.insert(0, str(Path(__file__).parent))

from imdb_recommender.config import AppConfig
from imdb_recommender.data_io import ingest_sources
from imdb_recommender.recommender_all_in_one import AllInOneRecommender


def main():
    """Run the All-in-One recommender demonstration."""

    print("🎬" + "=" * 70)
    print("🎬 ALL-IN-ONE FOUR-STAGE IMDb RECOMMENDER DEMO")
    print("🎬" + "=" * 70)
    print()

    # Load configuration and data
    print("📚 Loading data...")
    try:
        cfg = AppConfig.from_file("config.toml")
        res = ingest_sources(cfg.ratings_csv_path, cfg.watchlist_path, cfg.data_dir)
        print(
            "   ✅ Loaded "
            f"{len(res.dataset.ratings)} ratings "
            f"and {len(res.dataset.watchlist)} watchlist items"
        )
        print(f"   ✅ Total catalog: {len(res.dataset.catalog)} unique titles")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Make sure config.toml exists with valid data paths")
        return

    # Initialize the recommender
    print("🤖 Initializing All-in-One Recommender...")
    recommender = AllInOneRecommender(res.dataset, random_seed=42)
    print("   ✅ Recommender initialized with four-stage architecture")
    print()

    # Run the four-stage recommendation process
    print("🚀 Running Four-Stage Recommendation Process...")
    print("   Stage 1: Feature Engineering (content, popularity, temporal)")
    print("   Stage 2: Exposure Modeling (P(exposed) heuristic)")
    print("   Stage 3: Preference Modeling (pairwise learning)")
    print("   Stage 4: Diversity Optimization (MMR re-ranking)")
    print()

    scores, explanations = recommender.score(
        seeds=[],  # All-in-one doesn't need seed movies
        user_weight=0.7,
        global_weight=0.3,
        exclude_rated=True,
    )

    print(f"✅ Generated personalized scores for {len(scores)} titles")
    print()

    # Display top recommendations with detailed explanations
    print("🏆 TOP 10 PERSONALIZED RECOMMENDATIONS")
    print("=" * 60)

    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

    for i, (const, score) in enumerate(sorted_recs, 1):
        # Find title details
        item_row = res.dataset.catalog[res.dataset.catalog["imdb_const"] == const]
        if len(item_row) > 0:
            item = item_row.iloc[0]
            title = item.get("title", "Unknown")
            year = item.get("year", "Unknown")
            genres = item.get("genres", "Unknown")
            rating = item.get("imdb_rating", "N/A")
            votes = item.get("num_votes", "N/A")
            explanation = explanations.get(const, "AI-powered recommendation")

            print(f"{i:2d}. {title} ({year})")
            print(f"    🎭 Genres: {genres}")
            print(
                f"    ⭐ IMDb: {rating}/10 ({votes:,} votes)"
                if isinstance(votes, int)
                else f"    ⭐ IMDb: {rating}/10 ({votes} votes)"
            )
            print(f"    📊 Score: {score:.3f}")
            print(f"    💡 Why: {explanation}")
            print(f"    🔗 https://www.imdb.com/title/{const}/")
            print()

    # Show evaluation metrics if enough data
    if len(res.dataset.ratings) >= 10:
        print("📊 PERFORMANCE EVALUATION")
        print("=" * 40)
        print("Running temporal split evaluation...")

        metrics = recommender.evaluate_temporal_split(test_size=0.2)

        if metrics:
            print(
                "   📈 Hits@10: "
                f"{metrics.get('hits_at_10', 0):.4f} "
                "(fraction of test items in top-10)"
            )
            print(f"   📈 NDCG@10: {metrics.get('ndcg_at_10', 0):.4f} (ranking quality score)")
            print(f"   📈 Diversity: {metrics.get('diversity', 0):.4f} (recommendation variety)")
        else:
            print("   ⚠️ Evaluation metrics not available")
        print()

    # Export recommendations to CSV
    print("💾 EXPORTING RECOMMENDATIONS")
    print("=" * 35)

    export_path = "all_in_one_demo_recommendations.csv"
    recommender.export_recommendations_csv(scores, export_path, top_k=25)

    print(f"   ✅ Exported top 25 recommendations to {export_path}")
    print("   📋 Includes detailed scores: personal, popularity, final")
    print()

    # Save the trained model
    print("🗄️ SAVING TRAINED MODEL")
    print("=" * 30)

    model_path = "all_in_one_demo_model.pkl"
    recommender.save_model(model_path)

    print(f"   ✅ Model saved to {model_path}")
    print("   🔧 Includes all trained components and hyperparameters")
    print()

    # Show technical details
    print("🔬 TECHNICAL DETAILS")
    print("=" * 25)
    print(f"   🧠 Feature Matrix: {recommender.feature_matrix.shape} (items × features)")
    print(f"   🌌 Latent Space: {recommender.latent_features.shape} (items × components)")
    print(f"   ⚖️ Personal Weight: {recommender.personal_weight}")
    print(f"   📊 Popularity Weight: {recommender.popularity_weight}")
    print(f"   🎲 MMR Diversity: {recommender.mmr_lambda}")
    print(f"   📉 Recency Decay: {recommender.recency_lambda}")
    print()

    # Final summary
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 35)
    print("   🎯 Four-stage ML recommendation system successfully demonstrated")
    print("   📊 Intelligent scoring with exposure bias and preference modeling")
    print("   🎲 Diversity optimization prevents over-specialization")
    print("   💡 Explainable recommendations with detailed reasoning")
    print("   💾 Model persistence for fast future inference")
    print()
    print("📖 For more details, see docs/ALL_IN_ONE_GUIDE.md")
    print("🚀 Ready for production use with 'imdbrec all-in-one' command!")
    print()


if __name__ == "__main__":
    main()
