#!/usr/bin/env python3
# Canonical location: scripts/analysis/analyze_elasticnet_results.py
# Expected inputs: results/elasticnet_cv_comprehensive.csv
"""
Compare ElasticNet Cross Validation results with SVD baseline.
"""

import pandas as pd


def load_and_analyze_results():
    """Load and analyze ElasticNet CV results."""

    print("📊 ElasticNet Cross Validation Analysis")
    print("=" * 50)

    # Load results
    results_df = pd.read_csv("results/elasticnet_cv_comprehensive.csv")
    print(f"Loaded {len(results_df)} hyperparameter combinations")

    # Best model
    best = results_df.iloc[0]
    print("\n🏆 Best ElasticNet Model:")
    print(f"   Alpha: {best['alpha']}")
    print(f"   L1 Ratio: {best['l1_ratio']}")
    print(f"   RMSE: {best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}")
    print(f"   R²: {best['mean_r2']:.4f} ± {best['std_r2']:.4f}")

    # Performance summary
    print("\n📈 Performance Distribution:")
    print(f"   Best RMSE: {results_df['mean_rmse'].min():.4f}")
    print(f"   Worst RMSE: {results_df['mean_rmse'].max():.4f}")
    print(f"   Mean RMSE: {results_df['mean_rmse'].mean():.4f}")
    print(f"   Best R²: {results_df['mean_r2'].max():.4f}")
    print(f"   Worst R²: {results_df['mean_r2'].min():.4f}")

    # Hyperparameter analysis
    print("\n🔧 Hyperparameter Insights:")

    # Best alpha values
    top_5_alphas = results_df.head(5)["alpha"].value_counts()
    alpha_freq = top_5_alphas.iloc[0]
    alpha_val = top_5_alphas.index[0]
    print(f"   Most frequent alpha in top 5: {alpha_val} (appears {alpha_freq} times)")

    # Best l1_ratio values
    top_5_l1_ratios = results_df.head(5)["l1_ratio"].value_counts()
    l1_freq = top_5_l1_ratios.iloc[0]
    l1_val = top_5_l1_ratios.index[0]
    print(f"   Most frequent l1_ratio in top 5: {l1_val} (appears {l1_freq} times)")

    # Comparison with known baselines
    print("\n🆚 Model Comparison:")
    elasticnet_rmse = best["mean_rmse"]
    svd_rmse = 1.618  # From previous documentation
    baseline_rmse = 1.533  # From our diagnostic (mean predictor)

    print(f"   ElasticNet RMSE: {elasticnet_rmse:.4f}")
    print(f"   SVD RMSE: {svd_rmse:.4f}")
    print(f"   Baseline (mean) RMSE: {baseline_rmse:.4f}")
    svd_improvement = (svd_rmse - elasticnet_rmse) / svd_rmse * 100
    baseline_improvement = (baseline_rmse - elasticnet_rmse) / baseline_rmse * 100
    print(f"   ElasticNet vs SVD: {svd_improvement:+.1f}% improvement")
    print(f"   ElasticNet vs Baseline: {baseline_improvement:+.1f}% improvement")

    # Regularization analysis
    print("\n🎯 Regularization Analysis:")
    (
        results_df.groupby("alpha")
        .agg({"mean_rmse": ["mean", "min"], "mean_r2": ["mean", "max"]})
        .round(4)
    )

    print("   Performance by Alpha:")
    for alpha in sorted(results_df["alpha"].unique()):
        alpha_data = results_df[results_df["alpha"] == alpha]
        best_rmse = alpha_data["mean_rmse"].min()
        best_r2 = alpha_data["mean_r2"].max()
        print(f"     α={alpha:5.3f}: Best RMSE={best_rmse:.4f}, Best R²={best_r2:.4f}")

    return results_df


def create_performance_summary():
    """Create a comprehensive performance summary."""

    summary = {
        "Model": ["ElasticNet (Best)", "SVD", "Baseline (Mean)"],
        "RMSE": [1.3859, 1.618, 1.533],
        "R²": [0.2344, "N/A", 0.0],
        "Features": ["106 Engineered", "Latent Factors", "None"],
        "Approach": ["Linear Regression", "Matrix Factorization", "Constant Predictor"],
        "Hyperparameters": ["α=0.1, l1=0.1", "Various", "None"],
    }

    summary_df = pd.DataFrame(summary)
    print("\n📊 Model Performance Summary:")
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    """Main analysis function."""

    # Load and analyze results
    load_and_analyze_results()

    # Create performance summary
    summary_df = create_performance_summary()

    # Save summary
    summary_df.to_csv("results/model_comparison_summary.csv", index=False)

    print("\n✅ Analysis complete!")
    print("   Detailed results: results/elasticnet_cv_comprehensive.csv")
    print("   Summary: results/model_comparison_summary.csv")
    print("   Full report: results/ELASTICNET_CV_SUMMARY.md")


if __name__ == "__main__":
    main()
