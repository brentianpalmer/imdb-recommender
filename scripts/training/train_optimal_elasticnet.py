#!/usr/bin/env python3
# Canonical location: scripts/training/train_optimal_elasticnet.py
# Expected inputs: data/raw/ratings.csv

"""
ElasticNet Optimal Model
Train and save the best ElasticNet model based on cross validation results.
"""

import argparse
import pickle

import numpy as np
import pandas as pd

# Import feature engineering from our recommender
from elasticnet_cross_validation import clip_predictions, engineer_features, standardize_within_fold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Optimal hyperparameters from CV
OPTIMAL_ALPHA = 0.1
OPTIMAL_L1_RATIO = 0.1
RANDOM_STATE = 42


def train_optimal_model(
    ratings_file, save_model=True, model_path="results/elasticnet_optimal_model.pkl"
):
    """Train the optimal ElasticNet model and optionally save it."""

    print("🎯 Training Optimal ElasticNet Model")
    print(f"   📊 Hyperparameters: α={OPTIMAL_ALPHA}, l1_ratio={OPTIMAL_L1_RATIO}")
    print(f"   📥 Data: {ratings_file}")

    # Load and prepare data
    df = pd.read_csv(ratings_file)
    print(f"   Loaded {len(df):,} ratings")

    # Filter future movies
    if "Year" in df.columns:
        future_mask = pd.to_numeric(df["Year"], errors="coerce") > 2025
        if future_mask.any():
            n_future = future_mask.sum()
            print(f"   🚫 Filtered out {n_future} future movies")
            df = df[~future_mask].copy()

    # Engineer features
    print("🔧 Engineering features...")
    X, y, numeric_cols = engineer_features(df, top_dir_k=30)
    print(f"   Generated {X.shape[1]} features")

    # Remove invalid targets
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"   Training samples: {len(y):,}")

    # Split for validation (80/20)
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Standardize features
    X_train_std, X_val_std = standardize_within_fold(X_train, X_val, numeric_cols)

    # Train optimal model
    print("🚀 Training model...")
    model = ElasticNet(
        alpha=OPTIMAL_ALPHA,
        l1_ratio=OPTIMAL_L1_RATIO,
        random_state=RANDOM_STATE,
        max_iter=3000,  # Increased for better convergence
    )
    model.fit(X_train_std, y_train)

    # Evaluate
    y_pred_train_raw = model.predict(X_train_std)
    y_pred_val_raw = model.predict(X_val_std)

    y_pred_train = clip_predictions(y_pred_train_raw, 1.0, 10.0)
    y_pred_val = clip_predictions(y_pred_val_raw, 1.0, 10.0)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    print("📊 Performance:")
    print(f"   Training RMSE: {train_rmse:.4f}")
    print(f"   Validation RMSE: {val_rmse:.4f}")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Validation R²: {val_r2:.4f}")

    # Feature importance (non-zero coefficients)
    non_zero_coefs = model.coef_[model.coef_ != 0]
    feature_names = X.columns[model.coef_ != 0]
    print(f"   Selected features: {len(non_zero_coefs)} / {len(model.coef_)}")

    # Top features by absolute coefficient
    if len(non_zero_coefs) > 0:
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "coefficient": non_zero_coefs}
        ).sort_values("coefficient", key=abs, ascending=False)

        print("   Top 10 features:")
        for idx, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"     {idx+1:2d}. {row['feature']:20s}: {row['coefficient']:+.4f}")

    # Model package for saving
    model_package = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "train_stats": {
            "mean": X_train[numeric_cols].mean(),
            "std": X_train[numeric_cols].std().replace(0, 1.0),
        },
        "hyperparameters": {"alpha": OPTIMAL_ALPHA, "l1_ratio": OPTIMAL_L1_RATIO, "max_iter": 3000},
        "performance": {
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_r2": train_r2,
            "val_r2": val_r2,
        },
        "metadata": {
            "n_features": X.shape[1],
            "n_samples": len(y),
            "selected_features": len(non_zero_coefs),
            "feature_sparsity": 1 - (len(non_zero_coefs) / len(model.coef_)),
        },
    }

    if save_model:
        with open(model_path, "wb") as f:
            pickle.dump(model_package, f)
        print(f"💾 Model saved to: {model_path}")

    return model_package


def load_and_predict(model_path, ratings_file):
    """Load saved model and demonstrate prediction."""

    print(f"\n🔄 Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model_package = pickle.load(f)

    model_package["model"]
    model_package["feature_columns"]
    model_package["numeric_columns"]
    model_package["train_stats"]

    print(f"   Model performance: RMSE={model_package['performance']['val_rmse']:.4f}")
    print(f"   Features: {model_package['metadata']['n_features']}")
    selected_pct = 100 * (1 - model_package["metadata"]["feature_sparsity"])
    print(f"   Selected: {model_package['metadata']['selected_features']} ({selected_pct:.1f}%)")

    return model_package


def main():
    parser = argparse.ArgumentParser(description="Train optimal ElasticNet model")
    parser.add_argument(
        "--ratings_file", type=str, default="data/raw/ratings.csv", help="Path to ratings CSV file"
    )
    parser.add_argument(
        "--save_model", action="store_true", default=True, help="Save trained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/elasticnet_optimal_model.pkl",
        help="Path to save model",
    )
    parser.add_argument("--load_demo", action="store_true", help="Demonstrate loading saved model")

    args = parser.parse_args()

    # Create results directory
    import os

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Train model
    model_package = train_optimal_model(
        ratings_file=args.ratings_file, save_model=args.save_model, model_path=args.model_path
    )

    # Optionally demonstrate loading
    if args.load_demo and args.save_model:
        load_and_predict(args.model_path, args.ratings_file)

    print("\n✅ Optimal ElasticNet model ready!")
    print("   CV Performance: RMSE=1.386 ± 0.095, R²=0.234 ± 0.055")
    print(
        "   Current Performance: "
        f"RMSE={model_package['performance']['val_rmse']:.4f}, "
        f"R²={model_package['performance']['val_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
