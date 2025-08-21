#!/usr/bin/env python3
# Canonical location: scripts/diagnostics/debug_elasticnet_cv.py
# Expected inputs: data/raw/ratings.csv
"""
Diagnostic script to debug ElasticNet cross validation issues.
"""

import numpy as np
import pandas as pd

# Import the feature engineering from our CV script
from elasticnet_cross_validation import engineer_features, sigmoid_scale, standardize_within_fold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def main():
    print("🔍 Diagnosing ElasticNet Cross Validation Issues")
    print("=" * 50)

    # Load data
    print("📥 Loading data...")
    df = pd.read_csv("data/raw/ratings.csv")
    print(f"   Loaded {len(df):,} ratings")

    # Basic data inspection
    if "Your Rating" in df.columns:
        ratings = pd.to_numeric(df["Your Rating"], errors="coerce")
        print(f"   Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
        print(f"   Rating mean: {ratings.mean():.2f}")
        print(f"   Rating std: {ratings.std():.2f}")

    # Feature engineering
    print("\n🔧 Feature Engineering...")
    X, y, numeric_cols = engineer_features(df, top_dir_k=30)
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Feature range - min: {X.min().min():.3f}, max: {X.max().max():.3f}")

    # Check for issues
    print("\n🔍 Checking for common issues...")

    # Check target distribution
    print(
        f"   Target stats: mean={y.mean():.2f}, std={y.std():.2f}, "
        f"range=[{y.min():.1f}, {y.max():.1f}]"
    )

    # Check for extreme values in features
    extreme_features = []
    for col in X.columns:
        if X[col].max() > 1000 or X[col].min() < -1000:
            extreme_features.append((col, X[col].min(), X[col].max()))

    if extreme_features:
        print("   ⚠️  Features with extreme values:")
        for col, min_val, max_val in extreme_features[:5]:  # Show top 5
            print(f"      {col}: {min_val:.2f} to {max_val:.2f}")

    # Check feature correlation with target
    print("\n📊 Feature-target correlations (top 10):")
    correlations = []
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations[:10]:
        print(f"   {col}: {corr:.3f}")

    # Test simple train/test split
    print("\n🧪 Testing simple train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Test without standardization first
    print("   Testing without standardization...")
    model_raw = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    model_raw.fit(X_train, y_train)
    y_pred_raw = model_raw.predict(X_test)
    y_pred_scaled = sigmoid_scale(y_pred_raw, 1.0, 10.0)

    rmse_raw = np.sqrt(mean_squared_error(y_test, y_pred_raw))
    r2_raw = r2_score(y_test, y_pred_raw)
    rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    r2_scaled = r2_score(y_test, y_pred_scaled)

    print(f"      Raw predictions: RMSE={rmse_raw:.3f}, R²={r2_raw:.3f}")
    print(f"      Raw pred range: {y_pred_raw.min():.2f} to {y_pred_raw.max():.2f}")
    print(f"      Scaled predictions: RMSE={rmse_scaled:.3f}, R²={r2_scaled:.3f}")
    print(f"      Scaled pred range: {y_pred_scaled.min():.2f} to {y_pred_scaled.max():.2f}")

    # Test with standardization
    print("   Testing with standardization...")
    X_train_std, X_test_std = standardize_within_fold(X_train, X_test, numeric_cols)

    model_std = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    model_std.fit(X_train_std, y_train)
    y_pred_std_raw = model_std.predict(X_test_std)
    y_pred_std_scaled = sigmoid_scale(y_pred_std_raw, 1.0, 10.0)

    rmse_std_raw = np.sqrt(mean_squared_error(y_test, y_pred_std_raw))
    r2_std_raw = r2_score(y_test, y_pred_std_raw)
    rmse_std_scaled = np.sqrt(mean_squared_error(y_test, y_pred_std_scaled))
    r2_std_scaled = r2_score(y_test, y_pred_std_scaled)

    print(f"      Standardized raw: RMSE={rmse_std_raw:.3f}, R²={r2_std_raw:.3f}")
    print(f"      Std raw pred range: {y_pred_std_raw.min():.2f} to {y_pred_std_raw.max():.2f}")
    print(f"      Standardized scaled: RMSE={rmse_std_scaled:.3f}, R²={r2_std_scaled:.3f}")
    print(
        f"      Std scaled pred range: {y_pred_std_scaled.min():.2f} to "
        f"{y_pred_std_scaled.max():.2f}"
    )

    # Baseline comparison
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"\n📊 Baseline (mean) RMSE: {baseline_rmse:.3f}")

    # Try different hyperparameters
    print("\n🔧 Testing different hyperparameters...")
    test_params = [
        (0.001, 0.1),
        (0.001, 0.5),
        (0.001, 0.9),
        (0.01, 0.1),
        (0.01, 0.5),
        (0.01, 0.9),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.1, 0.9),
    ]

    best_rmse = float("inf")
    best_params = None

    for alpha, l1_ratio in test_params:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=2000)
        model.fit(X_train_std, y_train)
        y_pred = sigmoid_scale(model.predict(X_test_std), 1.0, 10.0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (alpha, l1_ratio)

        print(f"   α={alpha:5.3f}, l1={l1_ratio:.1f}: RMSE={rmse:.3f}, R²={r2:.3f}")

    print(f"\n🏆 Best params: α={best_params[0]}, l1_ratio={best_params[1]}, RMSE={best_rmse:.3f}")


if __name__ == "__main__":
    main()
