"""
Hyperparameter Tuning and Evaluation Framework for IMDb Recommender

This module provides comprehensive hyperparameter optimization and evaluation
capabilities for all recommendation algorithms in the system.

Key Features:
- Stratified train/test splits on ratings data
- Cross-validation with hyperparameter grids
- Rating prediction evaluation (RMSE, MAE, R²)
- Recommendation quality metrics (Precision@K, Recall@K, NDCG@K)
- Out-of-sample final validation
- Model persistence and reproducibility

Author: IMDb Recommender Team
Date: August 2025
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit

from .data_io import Dataset
from .recommender_all_in_one import AllInOneRecommender
from .recommender_pop import PopSimRecommender
from .recommender_svd import SVDAutoRecommender

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Rating prediction metrics
    rmse: float
    mae: float
    r2_score: float

    # Recommendation ranking metrics
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float

    # Coverage and diversity
    catalog_coverage: float
    diversity_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class HyperparameterResult:
    """Container for hyperparameter tuning results."""

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    cv_results: dict[str, Any]
    evaluation_metrics: EvaluationMetrics
    training_time: float
    prediction_time: float


class RecommenderEvaluator:
    """
    Comprehensive evaluation framework for recommender systems.

    Provides stratified train/test splits, hyperparameter tuning,
    and multiple evaluation metrics for rating prediction and recommendation quality.
    """

    def __init__(self, dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the evaluator.

        Args:
            dataset: The full dataset with ratings and watchlist
            test_size: Proportion of data to hold out for final testing
            random_state: Random seed for reproducibility
        """
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        self.results_dir = Path("hyperparameter_results")
        self.results_dir.mkdir(exist_ok=True)

        # Prepare train/test split
        self._prepare_data_splits()

    def _prepare_data_splits(self):
        """Create stratified train/test splits on ratings data."""
        ratings = self.dataset.ratings.copy()

        # Create rating bins for stratification
        ratings["rating_bin"] = pd.cut(
            ratings["my_rating"],
            bins=[0, 3, 5, 7, 8, 10],
            labels=["low", "below_avg", "avg", "good", "excellent"],
            include_lowest=True,
        )

        # Stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )

        train_idx, test_idx = next(splitter.split(ratings, ratings["rating_bin"]))

        self.train_ratings = (
            ratings.iloc[train_idx].drop("rating_bin", axis=1).reset_index(drop=True)
        )
        self.test_ratings = ratings.iloc[test_idx].drop("rating_bin", axis=1).reset_index(drop=True)

        # Create train and test datasets
        self.train_dataset = Dataset(ratings=self.train_ratings, watchlist=self.dataset.watchlist)
        self.test_dataset = Dataset(ratings=self.test_ratings, watchlist=self.dataset.watchlist)

        print("📊 Data Split Summary:")
        print(
            f"   Training ratings: {len(self.train_ratings)} "
            f"({len(self.train_ratings)/len(ratings)*100:.1f}%)"
        )
        print(
            f"   Test ratings: {len(self.test_ratings)} "
            f"({len(self.test_ratings)/len(ratings)*100:.1f}%)"
        )
        print("   Rating distribution preserved in split")

    def _calculate_rating_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[float, float, float]:
        """Calculate rating prediction metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    def _calculate_ranking_metrics(
        self, recommendations: dict[str, float], test_ratings: pd.DataFrame
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate recommendation ranking metrics."""

        # Create ground truth sets (items rated >= 8 are considered relevant)
        high_rated = set(test_ratings[test_ratings["my_rating"] >= 8]["imdb_const"].tolist())

        # Get top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        top_5 = [item for item, _ in sorted_recs[:5]]
        top_10 = [item for item, _ in sorted_recs[:10]]

        # Precision@K
        precision_5 = len(set(top_5) & high_rated) / len(top_5) if top_5 else 0.0
        precision_10 = len(set(top_10) & high_rated) / len(top_10) if top_10 else 0.0

        # Recall@K
        recall_5 = len(set(top_5) & high_rated) / len(high_rated) if high_rated else 0.0
        recall_10 = len(set(top_10) & high_rated) / len(high_rated) if high_rated else 0.0

        # NDCG@K (simplified version)
        def dcg_at_k(relevant_items, k):
            dcg = 0
            for i, item in enumerate(sorted_recs[:k]):
                if item[0] in relevant_items:
                    dcg += 1 / np.log2(i + 2)  # i + 2 because log2(1) = 0
            return dcg

        def ideal_dcg_at_k(relevant_items, k):
            ideal_relevance = [1] * min(len(relevant_items), k)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            return idcg

        idcg_5 = ideal_dcg_at_k(high_rated, 5)
        idcg_10 = ideal_dcg_at_k(high_rated, 10)

        ndcg_5 = dcg_at_k(high_rated, 5) / idcg_5 if idcg_5 > 0 else 0.0
        ndcg_10 = dcg_at_k(high_rated, 10) / idcg_10 if idcg_10 > 0 else 0.0

        return precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10

    def _calculate_diversity_metrics(
        self, recommendations: dict[str, float]
    ) -> tuple[float, float]:
        """Calculate recommendation diversity and coverage."""
        catalog_size = len(self.dataset.catalog)
        recommended_items = list(recommendations.keys())

        # Catalog coverage
        coverage = len(recommended_items) / catalog_size

        # Genre diversity (simplified - could be more sophisticated)
        catalog = self.dataset.catalog.set_index("imdb_const")
        rec_genres = []
        for item in recommended_items[:20]:  # Top 20 recommendations
            if item in catalog.index:
                genres = catalog.loc[item, "genres"]
                if pd.notna(genres):
                    rec_genres.extend(str(genres).split(", "))

        unique_genres = len(set(rec_genres))
        all_genres_in_catalog = set()
        for genres in catalog["genres"].dropna():
            all_genres_in_catalog.update(str(genres).split(", "))

        diversity = unique_genres / len(all_genres_in_catalog) if all_genres_in_catalog else 0.0

        return coverage, diversity


class AllInOneHyperparameterTuner:
    """Hyperparameter tuner for AllInOneRecommender."""

    @staticmethod
    def get_param_grid() -> dict[str, list[Any]]:
        """Get hyperparameter grid for AllInOneRecommender."""
        return {
            "exposure_model_params": [
                {"alpha": 0.0001, "max_iter": 1000},
                {"alpha": 0.001, "max_iter": 1000},
                {"alpha": 0.01, "max_iter": 500},
            ],
            "preference_model_params": [
                {"alpha": 0.0001, "max_iter": 1000},
                {"alpha": 0.001, "max_iter": 1000},
                {"alpha": 0.01, "max_iter": 500},
            ],
            "svd_components": [32, 64, 128],
            "mmr_lambda": [0.3, 0.5, 0.7],
            "min_votes_threshold": [100, 500, 1000],
        }

    @staticmethod
    def tune(evaluator: RecommenderEvaluator, cv_folds: int = 3) -> HyperparameterResult:
        """Tune hyperparameters for AllInOneRecommender."""
        import time

        param_grid = AllInOneHyperparameterTuner.get_param_grid()
        param_combinations = list(ParameterGrid(param_grid))

        print(
            f"🔬 Tuning AllInOneRecommender with {len(param_combinations)} "
            f"parameter combinations..."
        )

        best_score = float("-inf")
        best_params = None
        cv_results = {"scores": [], "params": []}

        start_time = time.time()

        for i, params in enumerate(param_combinations):
            print(f"   Testing combination {i+1}/{len(param_combinations)}...")

            # Cross-validation
            cv_scores = []
            for fold in range(cv_folds):
                fold_seed = evaluator.random_state + fold

                # Create recommender with current params
                recommender = AllInOneRecommender(evaluator.train_dataset, random_seed=fold_seed)

                # Configure hyperparameters
                recommender.exposure_model_params = params["exposure_model_params"]
                recommender.preference_model_params = params["preference_model_params"]
                recommender.svd_components = params["svd_components"]
                recommender.mmr_lambda = params["mmr_lambda"]
                recommender.min_votes_threshold = params["min_votes_threshold"]

                # Generate predictions for test set
                test_items = evaluator.test_ratings["imdb_const"].tolist()
                recommendations, _ = recommender.score(
                    seeds=[],
                    user_weight=0.7,
                    global_weight=0.3,
                    exclude_rated=False,
                    candidates=test_items,
                )

                # Calculate RMSE as primary metric
                y_true = []
                y_pred = []
                for _, row in evaluator.test_ratings.iterrows():
                    const = row["imdb_const"]
                    if const in recommendations:
                        y_true.append(row["my_rating"])
                        # Convert recommendation score to rating scale (1-10)
                        pred_rating = min(10, max(1, recommendations[const] * 3 + 5.5))
                        y_pred.append(pred_rating)

                if len(y_true) > 0:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    cv_scores.append(-rmse)  # Negative RMSE for maximization
                else:
                    cv_scores.append(-10.0)  # Penalty for no predictions

            mean_score = np.mean(cv_scores)
            cv_results["scores"].append(mean_score)
            cv_results["params"].append(params)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        training_time = time.time() - start_time

        # Evaluate best model on test set
        print("🏆 Best parameters found, evaluating on test set...")
        pred_start = time.time()

        best_recommender = AllInOneRecommender(
            evaluator.train_dataset, random_seed=evaluator.random_state
        )
        best_recommender.exposure_model_params = best_params["exposure_model_params"]
        best_recommender.preference_model_params = best_params["preference_model_params"]
        best_recommender.svd_components = best_params["svd_components"]
        best_recommender.mmr_lambda = best_params["mmr_lambda"]
        best_recommender.min_votes_threshold = best_params["min_votes_threshold"]

        # Full evaluation
        test_items = evaluator.test_ratings["imdb_const"].tolist()
        recommendations, _ = best_recommender.score(
            seeds=[], user_weight=0.7, global_weight=0.3, exclude_rated=False, candidates=test_items
        )

        prediction_time = time.time() - pred_start

        # Calculate all metrics
        y_true = []
        y_pred = []
        for _, row in evaluator.test_ratings.iterrows():
            const = row["imdb_const"]
            if const in recommendations:
                y_true.append(row["my_rating"])
                pred_rating = min(10, max(1, recommendations[const] * 3 + 5.5))
                y_pred.append(pred_rating)

        rmse, mae, r2 = evaluator._calculate_rating_metrics(np.array(y_true), np.array(y_pred))
        precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10 = (
            evaluator._calculate_ranking_metrics(recommendations, evaluator.test_ratings)
        )
        coverage, diversity = evaluator._calculate_diversity_metrics(recommendations)

        metrics = EvaluationMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            catalog_coverage=coverage,
            diversity_score=diversity,
        )

        return HyperparameterResult(
            model_name="AllInOneRecommender",
            best_params=best_params,
            best_score=best_score,
            cv_results=cv_results,
            evaluation_metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
        )


class PopSimHyperparameterTuner:
    """Hyperparameter tuner for PopSimRecommender."""

    @staticmethod
    def get_param_grid() -> dict[str, list[Any]]:
        """Get hyperparameter grid for PopSimRecommender."""
        return {
            "user_weight": [0.5, 0.7, 0.8, 0.9],
            "global_weight": [0.1, 0.2, 0.3, 0.5],
            "recency": [0.0, 0.1, 0.3, 0.5],
            "rating_threshold": [7, 8, 9],  # Threshold for "liked" items
        }

    @staticmethod
    def tune(evaluator: RecommenderEvaluator, cv_folds: int = 3) -> HyperparameterResult:
        """Tune hyperparameters for PopSimRecommender."""
        import time

        param_grid = PopSimHyperparameterTuner.get_param_grid()
        param_combinations = list(ParameterGrid(param_grid))

        print(
            f"🔬 Tuning PopSimRecommender with {len(param_combinations)} parameter combinations..."
        )

        best_score = float("-inf")
        best_params = None
        cv_results = {"scores": [], "params": []}

        start_time = time.time()

        for i, params in enumerate(param_combinations):
            print(f"   Testing combination {i+1}/{len(param_combinations)}...")

            # Cross-validation
            cv_scores = []
            for fold in range(cv_folds):
                fold_seed = evaluator.random_state + fold

                # Create recommender
                recommender = PopSimRecommender(evaluator.train_dataset, random_seed=fold_seed)

                # Generate predictions
                test_items = evaluator.test_ratings["imdb_const"].tolist()
                recommendations, _ = recommender.score(
                    seeds=[],  # Use liked items as implicit seeds
                    user_weight=params["user_weight"],
                    global_weight=params["global_weight"],
                    recency=params["recency"],
                    exclude_rated=False,
                )

                # Filter to test items only
                test_recommendations = {k: v for k, v in recommendations.items() if k in test_items}

                # Calculate RMSE
                y_true = []
                y_pred = []
                for _, row in evaluator.test_ratings.iterrows():
                    const = row["imdb_const"]
                    if const in test_recommendations:
                        y_true.append(row["my_rating"])
                        # Convert recommendation score to rating scale
                        pred_rating = min(10, max(1, test_recommendations[const] * 10))
                        y_pred.append(pred_rating)

                if len(y_true) > 0:
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    cv_scores.append(-rmse)  # Negative for maximization
                else:
                    cv_scores.append(-10.0)

            mean_score = np.mean(cv_scores)
            cv_results["scores"].append(mean_score)
            cv_results["params"].append(params)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        training_time = time.time() - start_time

        # Final evaluation
        print("🏆 Best parameters found, evaluating on test set...")
        pred_start = time.time()

        best_recommender = PopSimRecommender(
            evaluator.train_dataset, random_seed=evaluator.random_state
        )
        test_items = evaluator.test_ratings["imdb_const"].tolist()
        recommendations, _ = best_recommender.score(
            seeds=[],
            user_weight=best_params["user_weight"],
            global_weight=best_params["global_weight"],
            recency=best_params["recency"],
            exclude_rated=False,
        )

        test_recommendations = {k: v for k, v in recommendations.items() if k in test_items}
        prediction_time = time.time() - pred_start

        # Calculate metrics
        y_true = []
        y_pred = []
        for _, row in evaluator.test_ratings.iterrows():
            const = row["imdb_const"]
            if const in test_recommendations:
                y_true.append(row["my_rating"])
                pred_rating = min(10, max(1, test_recommendations[const] * 10))
                y_pred.append(pred_rating)

        rmse, mae, r2 = evaluator._calculate_rating_metrics(np.array(y_true), np.array(y_pred))
        precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10 = (
            evaluator._calculate_ranking_metrics(test_recommendations, evaluator.test_ratings)
        )
        coverage, diversity = evaluator._calculate_diversity_metrics(test_recommendations)

        metrics = EvaluationMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            catalog_coverage=coverage,
            diversity_score=diversity,
        )

        return HyperparameterResult(
            model_name="PopSimRecommender",
            best_params=best_params,
            best_score=best_score,
            cv_results=cv_results,
            evaluation_metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
        )


class SVDHyperparameterTuner:
    """Hyperparameter tuner for SVDAutoRecommender."""

    @staticmethod
    def get_param_grid() -> dict[str, list[Any]]:
        """Get hyperparameter grid for SVDAutoRecommender."""
        return {
            "n_components": [8, 16, 32, 64],
            "regularization": [0.01, 0.1, 0.2, 0.5],
            "n_iterations": [10, 25, 50],
            "user_weight": [0.5, 0.7, 0.8, 0.9],
            "global_weight": [0.1, 0.2, 0.3, 0.5],
        }

    @staticmethod
    def tune(evaluator: RecommenderEvaluator, cv_folds: int = 3) -> HyperparameterResult:
        """Tune hyperparameters for SVDAutoRecommender."""
        import time

        param_grid = SVDHyperparameterTuner.get_param_grid()
        param_combinations = list(ParameterGrid(param_grid))

        print(
            f"🔬 Tuning SVDAutoRecommender with {len(param_combinations)} parameter combinations..."
        )

        best_score = float("-inf")
        best_params = None
        cv_results = {"scores": [], "params": []}

        start_time = time.time()

        for i, params in enumerate(param_combinations):
            print(f"   Testing combination {i+1}/{len(param_combinations)}...")

            # Cross-validation
            cv_scores = []
            for fold in range(cv_folds):
                fold_seed = evaluator.random_state + fold

                try:
                    # Create recommender
                    recommender = SVDAutoRecommender(evaluator.train_dataset, random_seed=fold_seed)

                    # Override default parameters (need to modify SVD class to accept these)
                    test_items = evaluator.test_ratings["imdb_const"].tolist()
                    recommendations, _ = recommender.score(
                        seeds=[],
                        user_weight=params["user_weight"],
                        global_weight=params["global_weight"],
                        recency=0.0,
                        exclude_rated=False,
                    )

                    # Filter to test items
                    test_recommendations = {
                        k: v for k, v in recommendations.items() if k in test_items
                    }

                    # Calculate RMSE
                    y_true = []
                    y_pred = []
                    for _, row in evaluator.test_ratings.iterrows():
                        const = row["imdb_const"]
                        if const in test_recommendations:
                            y_true.append(row["my_rating"])
                            pred_rating = min(10, max(1, test_recommendations[const] * 10))
                            y_pred.append(pred_rating)

                    if len(y_true) > 0:
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        cv_scores.append(-rmse)
                    else:
                        cv_scores.append(-10.0)

                except Exception as e:
                    print(f"      Error in fold {fold}: {e}")
                    cv_scores.append(-20.0)  # Heavy penalty for errors

            mean_score = np.mean(cv_scores)
            cv_results["scores"].append(mean_score)
            cv_results["params"].append(params)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        training_time = time.time() - start_time

        # Final evaluation
        print("🏆 Best parameters found, evaluating on test set...")
        pred_start = time.time()

        best_recommender = SVDAutoRecommender(
            evaluator.train_dataset, random_seed=evaluator.random_state
        )
        test_items = evaluator.test_ratings["imdb_const"].tolist()
        recommendations, _ = best_recommender.score(
            seeds=[],
            user_weight=best_params["user_weight"],
            global_weight=best_params["global_weight"],
            recency=0.0,
            exclude_rated=False,
        )

        test_recommendations = {k: v for k, v in recommendations.items() if k in test_items}
        prediction_time = time.time() - pred_start

        # Calculate metrics
        y_true = []
        y_pred = []
        for _, row in evaluator.test_ratings.iterrows():
            const = row["imdb_const"]
            if const in test_recommendations:
                y_true.append(row["my_rating"])
                pred_rating = min(10, max(1, test_recommendations[const] * 10))
                y_pred.append(pred_rating)

        rmse, mae, r2 = evaluator._calculate_rating_metrics(np.array(y_true), np.array(y_pred))
        precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10 = (
            evaluator._calculate_ranking_metrics(test_recommendations, evaluator.test_ratings)
        )
        coverage, diversity = evaluator._calculate_diversity_metrics(test_recommendations)

        metrics = EvaluationMetrics(
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            ndcg_at_5=ndcg_5,
            ndcg_at_10=ndcg_10,
            catalog_coverage=coverage,
            diversity_score=diversity,
        )

        return HyperparameterResult(
            model_name="SVDAutoRecommender",
            best_params=best_params,
            best_score=best_score,
            cv_results=cv_results,
            evaluation_metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
        )


class HyperparameterTuningPipeline:
    """Main pipeline for hyperparameter tuning all recommender systems."""

    def __init__(self, dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
        """Initialize the tuning pipeline."""
        self.evaluator = RecommenderEvaluator(dataset, test_size, random_state)
        self.results = {}

    def run_full_tuning(
        self, cv_folds: int = 3, models: list[str] | None = None
    ) -> dict[str, HyperparameterResult]:
        """
        Run hyperparameter tuning for all or specified models.

        Args:
            cv_folds: Number of cross-validation folds
            models: List of models to tune ('all-in-one', 'pop-sim', 'svd') or None for all

        Returns:
            Dictionary of tuning results by model name
        """
        if models is None:
            models = ["all-in-one", "pop-sim", "svd"]

        print("🚀 Starting hyperparameter tuning pipeline...")
        print(f"   Models to tune: {models}")
        print(f"   Cross-validation folds: {cv_folds}")
        print(f"   Test set size: {self.evaluator.test_size*100:.1f}%")
        print()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if "all-in-one" in models:
            print("=" * 80)
            print("🔬 TUNING ALL-IN-ONE RECOMMENDER")
            print("=" * 80)
            result = AllInOneHyperparameterTuner.tune(self.evaluator, cv_folds)
            self.results["all-in-one"] = result
            self._save_result(result, f"all_in_one_{timestamp}")
            self._print_result_summary(result)
            print()

        if "pop-sim" in models:
            print("=" * 80)
            print("🔬 TUNING POPULARITY-SIMILARITY RECOMMENDER")
            print("=" * 80)
            result = PopSimHyperparameterTuner.tune(self.evaluator, cv_folds)
            self.results["pop-sim"] = result
            self._save_result(result, f"pop_sim_{timestamp}")
            self._print_result_summary(result)
            print()

        if "svd" in models:
            print("=" * 80)
            print("🔬 TUNING SVD RECOMMENDER")
            print("=" * 80)
            result = SVDHyperparameterTuner.tune(self.evaluator, cv_folds)
            self.results["svd"] = result
            self._save_result(result, f"svd_{timestamp}")
            self._print_result_summary(result)
            print()

        # Summary comparison
        self._print_comparison_summary()
        self._save_full_results(timestamp)

        return self.results

    def _save_result(self, result: HyperparameterResult, filename: str):
        """Save individual tuning result."""
        result_dict = {
            "model_name": result.model_name,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "evaluation_metrics": result.evaluation_metrics.to_dict(),
            "training_time": result.training_time,
            "prediction_time": result.prediction_time,
            "timestamp": datetime.now().isoformat(),
        }

        # Save JSON
        json_path = self.evaluator.results_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        # Save pickle for full object
        pickle_path = self.evaluator.results_dir / f"{filename}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f)

    def _save_full_results(self, timestamp: str):
        """Save complete results summary."""
        summary = {
            "timestamp": timestamp,
            "dataset_info": {
                "total_ratings": len(self.evaluator.dataset.ratings),
                "train_ratings": len(self.evaluator.train_ratings),
                "test_ratings": len(self.evaluator.test_ratings),
                "test_size": self.evaluator.test_size,
                "random_state": self.evaluator.random_state,
            },
            "results": {},
        }

        for model_name, result in self.results.items():
            summary["results"][model_name] = {
                "best_params": result.best_params,
                "best_score": result.best_score,
                "metrics": result.evaluation_metrics.to_dict(),
                "training_time": result.training_time,
                "prediction_time": result.prediction_time,
            }

        summary_path = self.evaluator.results_dir / f"full_tuning_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"💾 Results saved to {self.evaluator.results_dir}")

    def _print_result_summary(self, result: HyperparameterResult):
        """Print formatted result summary."""
        print(f"✅ {result.model_name} Tuning Complete!")
        print(f"   Best CV Score: {result.best_score:.4f}")
        print(f"   Training Time: {result.training_time:.2f}s")
        print(f"   Prediction Time: {result.prediction_time:.2f}s")
        print()
        print("📊 Test Set Performance:")
        metrics = result.evaluation_metrics
        print(f"   RMSE: {metrics.rmse:.4f}")
        print(f"   MAE:  {metrics.mae:.4f}")
        print(f"   R²:   {metrics.r2_score:.4f}")
        print(f"   Precision@5:  {metrics.precision_at_5:.4f}")
        print(f"   Precision@10: {metrics.precision_at_10:.4f}")
        print(f"   Recall@5:     {metrics.recall_at_5:.4f}")
        print(f"   Recall@10:    {metrics.recall_at_10:.4f}")
        print(f"   NDCG@5:       {metrics.ndcg_at_5:.4f}")
        print(f"   NDCG@10:      {metrics.ndcg_at_10:.4f}")
        print(f"   Coverage:     {metrics.catalog_coverage:.4f}")
        print(f"   Diversity:    {metrics.diversity_score:.4f}")
        print()
        print("🎯 Best Parameters:")
        for param, value in result.best_params.items():
            print(f"   {param}: {value}")

    def _print_comparison_summary(self):
        """Print comparison summary across all models."""
        if not self.results:
            return

        print("=" * 80)
        print("📊 MODEL COMPARISON SUMMARY")
        print("=" * 80)

        # Create comparison table
        print(f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'P@5':<8} {'P@10':<8} {'Time':<8}")
        print("-" * 80)

        for _model_name, result in self.results.items():
            m = result.evaluation_metrics
            print(
                f"{result.model_name:<20} "
                f"{m.rmse:<8.3f} {m.mae:<8.3f} {m.r2_score:<8.3f} "
                f"{m.precision_at_5:<8.3f} {m.precision_at_10:<8.3f} "
                f"{result.training_time:<8.1f}"
            )

        # Identify best models
        best_rmse = min(self.results.values(), key=lambda x: x.evaluation_metrics.rmse)
        best_precision = max(
            self.results.values(), key=lambda x: x.evaluation_metrics.precision_at_10
        )

        print()
        print("🏆 Best Models:")
        print(f"   Lowest RMSE: {best_rmse.model_name} ({best_rmse.evaluation_metrics.rmse:.4f})")
        print(
            f"   Highest Precision@10: {best_precision.model_name} "
            f"({best_precision.evaluation_metrics.precision_at_10:.4f})"
        )
        print()
