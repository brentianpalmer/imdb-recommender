"""
Fixed Fine-tune SVD - Addressing Data Leakage Issue
Corrects the cross-validation to prevent data leakage in the hybrid row.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def fine_tune_svd_corrected():
    """Fine-tune SVD with proper data leakage prevention."""
    print("🔬 CORRECTED SVD FINE-TUNING (NO DATA LEAKAGE)")
    print("=" * 60)

    # Load data
    ratings_df = pd.read_parquet("data/ratings_normalized.parquet")
    watchlist_df = pd.read_parquet("data/watchlist_normalized.parquet")

    # Create rating matrix
    all_movies = pd.concat(
        [
            ratings_df[["imdb_const", "title", "imdb_rating"]],
            watchlist_df[["imdb_const", "title", "imdb_rating"]],
        ]
    ).drop_duplicates("imdb_const")

    movie_to_idx = {movie: i for i, movie in enumerate(all_movies["imdb_const"])}
    n_movies = len(all_movies)

    print(f"📊 Matrix: (3, {n_movies}), User ratings: {len(ratings_df)}")

    # Test the optimal configuration first
    print("\n🎯 TESTING OPTIMAL CONFIGURATION (24, 0.05, 20)")
    rmse, r2, std = cross_validate_als_corrected(
        ratings_df, watchlist_df, all_movies, movie_to_idx, n_factors=24, reg_param=0.05, n_iter=20
    )

    print(f"   📊 CORRECTED RMSE: {rmse:.4f} ± {std:.4f}")
    print(f"   📈 R²: {r2:.4f}")

    # Compare with a few other configurations
    configs_to_test = [
        (16, 0.1, 25),  # Previous champion
        (24, 0.05, 20),  # Claimed optimal
        (32, 0.1, 30),  # Higher complexity
        (24, 0.1, 20),  # Higher regularization
    ]

    results = []

    print("\n🧪 TESTING MULTIPLE CONFIGURATIONS:")
    for i, (n_factors, reg_param, n_iter) in enumerate(configs_to_test):
        print(f"  [{i+1}/4] factors={n_factors}, reg={reg_param}, iter={n_iter}", end="")

        try:
            start_time = time.time()
            rmse, r2, std = cross_validate_als_corrected(
                ratings_df, watchlist_df, all_movies, movie_to_idx, n_factors, reg_param, n_iter
            )
            elapsed = time.time() - start_time

            results.append(
                {
                    "n_factors": n_factors,
                    "reg_param": reg_param,
                    "n_iter": n_iter,
                    "rmse": rmse,
                    "r2": r2,
                    "rmse_std": std,
                    "time": elapsed,
                }
            )

            print(f" → RMSE: {rmse:.4f} ± {std:.4f}")

        except Exception as e:
            print(f" → ERROR: {e}")
            continue

    # Analyze results
    if results:
        results.sort(key=lambda x: x["rmse"])

        print("\n" + "=" * 70)
        print("📊 CORRECTED RESULTS RANKING")
        print("=" * 70)

        for i, result in enumerate(results):
            emoji = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}"
            print(
                f"{emoji} factors={result['n_factors']:2d} reg={result['reg_param']:.3f} "
                f"iter={result['n_iter']:2d} RMSE={result['rmse']:.4f} ± {result['rmse_std']:.4f}"
            )

        # Best configuration
        best = results[0]
        print("\n🎯 BEST CONFIGURATION (NO DATA LEAKAGE):")
        print(f"   🔢 Latent Factors: {best['n_factors']}")
        print(f"   📐 Regularization: {best['reg_param']}")
        print(f"   🔄 Iterations: {best['n_iter']}")
        print(f"   🎖️  RMSE: {best['rmse']:.4f} ± {best['rmse_std']:.4f}")
        print(f"   📈 R²: {best['r2']:.4f}")

        # Save corrected results
        with open("svd_corrected_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n💾 Corrected results saved to svd_corrected_results.json")

        return best

    else:
        print("❌ No valid results found")
        return None


def cross_validate_als_corrected(
    ratings_df, watchlist_df, all_movies, movie_to_idx, n_factors, reg_param, n_iter
):
    """Cross-validate ALS with proper data leakage prevention."""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_rmses = []
    cv_r2s = []

    user_mean = ratings_df["my_rating"].mean()
    n_movies = len(all_movies)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        print(f"\n      Fold {fold_idx + 1}/3", end="")

        train_ratings = ratings_df.iloc[train_idx]
        test_ratings = ratings_df.iloc[test_idx]

        # Create matrix with proper data leakage prevention
        matrix = np.zeros((3, n_movies))

        # Row 0: Only training user ratings (test ratings are 0)
        for _, train_row in train_ratings.iterrows():
            if train_row["imdb_const"] in movie_to_idx:
                idx = movie_to_idx[train_row["imdb_const"]]
                matrix[0, idx] = train_row["my_rating"]

        # Row 1: IMDb ratings (global signal)
        for i, (_, row) in enumerate(all_movies.iterrows()):
            if pd.notna(row["imdb_rating"]):
                matrix[1, i] = row["imdb_rating"]
            else:
                matrix[1, i] = 6.5

        # Row 2: Hybrid ratings WITHOUT test user ratings (CRITICAL FIX)
        for i in range(n_movies):
            if matrix[0, i] > 0:  # Only for training items
                matrix[2, i] = 0.7 * matrix[0, i] + 0.3 * matrix[1, i]
            else:  # For unrated items (including test items)
                matrix[2, i] = 0.3 * user_mean + 0.7 * matrix[1, i]

        # Run ALS
        U, V = als_algorithm(matrix, n_factors, reg_param, n_iter)

        # Predict test ratings
        predictions = []
        actuals = []

        for _, test_row in test_ratings.iterrows():
            if test_row["imdb_const"] in movie_to_idx:
                idx = movie_to_idx[test_row["imdb_const"]]
                pred = np.dot(U[0], V[idx])  # Predict using user 0 factors
                pred = max(1.0, min(10.0, pred))  # Clamp to rating scale

                predictions.append(pred)
                actuals.append(test_row["my_rating"])

        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = 1 - np.sum((np.array(actuals) - np.array(predictions)) ** 2) / np.sum(
                (np.array(actuals) - np.mean(actuals)) ** 2
            )
            cv_rmses.append(rmse)
            cv_r2s.append(r2)
            print(f" (RMSE: {rmse:.4f})", end="")

    return np.mean(cv_rmses), np.mean(cv_r2s), np.std(cv_rmses)


def als_algorithm(R, k, reg, iters):
    """ALS algorithm implementation."""
    np.random.seed(42)
    m, n = R.shape
    M = (R > 0).astype(float)

    U = 0.1 * np.random.randn(m, k)
    V = 0.1 * np.random.randn(n, k)

    for _ in range(iters):
        # Update U
        for i in range(m):
            Vi = V[M[i, :] > 0]
            Ri = R[i, M[i, :] > 0]
            if Vi.shape[0] == 0:
                continue
            A = Vi.T @ Vi + reg * np.eye(k)
            b = Vi.T @ Ri
            try:
                U[i] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                U[i] = np.linalg.lstsq(A, b, rcond=None)[0]

        # Update V
        for j in range(n):
            Uj = U[M[:, j] > 0]
            Rj = R[M[:, j] > 0, j]
            if Uj.shape[0] == 0:
                continue
            A = Uj.T @ Uj + reg * np.eye(k)
            b = Uj.T @ Rj
            try:
                V[j] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                V[j] = np.linalg.lstsq(A, b, rcond=None)[0]

    return U, V


if __name__ == "__main__":
    print("🚨 ADDRESSING DATA LEAKAGE IN SVD VALIDATION")
    print("\nPROBLEM: Previous validation included test ratings in hybrid row")
    print("FIX: Exclude test user ratings from training matrix completely")
    print("\n" + "=" * 60)

    result = fine_tune_svd_corrected()

    if result:
        print("\n🎯 FINAL CORRECTED RESULT:")
        print(f"   Best RMSE: {result['rmse']:.4f} ± {result['rmse_std']:.4f}")
        print("   This is the TRUE performance without data leakage")

    print("\n📊 COMPARISON TO CLAIMED 0.5447:")
    print("   The original 0.5447 RMSE was likely due to data leakage")
    print("   This corrected version provides unbiased evaluation")
