"""
Fine-tune the Optimal SVD Configuration
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import json
import time


def fine_tune_optimal_svd():
    """Fine-tune the ALS SVD with different hyperparameters."""
    print("🔬 FINE-TUNING OPTIMAL SVD CONFIGURATION")
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

    matrix = np.zeros((3, n_movies))

    # Fill matrix
    for _, row in ratings_df.iterrows():
        if row["imdb_const"] in movie_to_idx:
            idx = movie_to_idx[row["imdb_const"]]
            matrix[0, idx] = row["my_rating"]

    for i, (_, row) in enumerate(all_movies.iterrows()):
        if pd.notna(row["imdb_rating"]):
            matrix[1, i] = row["imdb_rating"]
        else:
            matrix[1, i] = 6.5

    user_mean = ratings_df["my_rating"].mean()
    for i in range(n_movies):
        if matrix[0, i] > 0:
            matrix[2, i] = 0.7 * matrix[0, i] + 0.3 * matrix[1, i]
        else:
            matrix[2, i] = 0.3 * user_mean + 0.7 * matrix[1, i]

    print(f"📊 Matrix: {matrix.shape}, User ratings: {np.sum(matrix[0] > 0)}")

    # Hyperparameter grid
    param_grid = {
        "n_factors": [24, 28, 32, 36, 40, 48],  # Around the optimal 32
        "reg_param": [0.05, 0.08, 0.1, 0.12, 0.15],  # Around 0.1
        "n_iter": [20, 25, 30, 40, 50],  # Iteration counts
    }

    results = []
    total_combinations = (
        len(param_grid["n_factors"]) * len(param_grid["reg_param"]) * len(param_grid["n_iter"])
    )

    print(f"🧪 Testing {total_combinations} parameter combinations...")

    combination_count = 0

    for n_factors in param_grid["n_factors"]:
        for reg_param in param_grid["reg_param"]:
            for n_iter in param_grid["n_iter"]:
                combination_count += 1

                print(
                    f"  [{combination_count}/{total_combinations}] factors={n_factors}, reg={reg_param}, iter={n_iter}",
                    end="",
                )

                try:
                    start_time = time.time()
                    rmse, r2, std = cross_validate_als(
                        matrix, ratings_df, movie_to_idx, n_factors, reg_param, n_iter
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
        print("📊 TOP 10 CONFIGURATIONS")
        print("=" * 70)

        current_best = 0.8283

        for i, result in enumerate(results[:10]):
            improvement = (current_best - result["rmse"]) / current_best * 100
            emoji = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}"

            print(
                f"{emoji} factors={result['n_factors']:2d} reg={result['reg_param']:.3f} iter={result['n_iter']:2d} "
                f"RMSE={result['rmse']:.4f} R²={result['r2']:.4f} (+{improvement:+.1f}%)"
            )

        # Best configuration
        best = results[0]
        print(f"\n🎯 OPTIMAL SVD CONFIGURATION:")
        print(f"   🔢 Latent Factors: {best['n_factors']}")
        print(f"   📐 Regularization: {best['reg_param']}")
        print(f"   🔄 Iterations: {best['n_iter']}")
        print(f"   🎖️  RMSE: {best['rmse']:.4f} ± {best['rmse_std']:.4f}")
        print(f"   📈 R²: {best['r2']:.4f}")

        improvement = (current_best - best["rmse"]) / current_best * 100
        print(f"   🚀 Improvement: {improvement:.1f}% better than current!")

        # Save results
        with open("svd_fine_tuning_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to svd_fine_tuning_results.json")

        return best

    else:
        print("❌ No valid results found")
        return None


def cross_validate_als(matrix, ratings_df, movie_to_idx, n_factors, reg_param, n_iter):
    """Cross-validate ALS algorithm."""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_rmses = []
    cv_r2s = []

    for train_idx, test_idx in kf.split(ratings_df):
        train_ratings = ratings_df.iloc[train_idx]
        test_ratings = ratings_df.iloc[test_idx]

        # Create train matrix
        train_matrix = matrix.copy()
        for _, test_row in test_ratings.iterrows():
            if test_row["imdb_const"] in movie_to_idx:
                idx = movie_to_idx[test_row["imdb_const"]]
                train_matrix[0, idx] = 0

        # Run ALS
        U, V = als_algorithm(train_matrix, n_factors, reg_param, n_iter)

        # Predict
        predictions = []
        actuals = []

        for _, test_row in test_ratings.iterrows():
            if test_row["imdb_const"] in movie_to_idx:
                idx = movie_to_idx[test_row["imdb_const"]]
                pred = np.dot(U[0], V[idx])
                pred = max(1.0, min(10.0, pred))

                predictions.append(pred)
                actuals.append(test_row["my_rating"])

        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            r2 = 1 - np.sum((np.array(actuals) - np.array(predictions)) ** 2) / np.sum(
                (np.array(actuals) - np.mean(actuals)) ** 2
            )
            cv_rmses.append(rmse)
            cv_r2s.append(r2)

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
            except:
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
            except:
                V[j] = np.linalg.lstsq(A, b, rcond=None)[0]

    return U, V


if __name__ == "__main__":
    fine_tune_optimal_svd()
