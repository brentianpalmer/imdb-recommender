"""
K-Fold Cross-Validation Framework for IMDb Recommender Systems

This module implements proper K-fold cross-validation with stratified splits
for robust evaluation of recommendation algorithms. Unlike simple train/test
splits, this utility rotates validation folds to ensure each data point is
used for validation exactly once across all folds.

Key Features:
- Stratified K-fold splits preserving rating distribution
- Temporal K-fold splits respecting chronological order
- Support for both rating prediction and recommendation evaluation
- Proper isolation of training/validation/test sets
- Reproducible cross-validation results

Author: IMDb Recommender Team
Date: August 2025
"""

from __future__ import annotations

import warnings
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from .data_io import Dataset
from .recommender_base import RecommenderAlgo

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class CVFoldResult:
    """Results from a single cross-validation fold."""

    fold_idx: int
    train_size: int
    val_size: int
    rmse: float
    mae: float
    r2: float
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float


@dataclass
class CrossValidationResult:
    """Aggregated results from K-fold cross-validation."""

    n_folds: int
    fold_results: list[CVFoldResult]

    # Rating prediction metrics
    mean_rmse: float
    std_rmse: float
    mean_mae: float
    std_mae: float
    mean_r2: float
    std_r2: float

    # Ranking metrics
    mean_precision_at_10: float
    std_precision_at_10: float
    mean_recall_at_10: float
    std_recall_at_10: float
    mean_ndcg_at_10: float
    std_ndcg_at_10: float


class StratifiedKFoldCV:
    """
    Stratified K-fold cross-validation for recommendation systems.

    Ensures each fold maintains the same rating distribution as the original dataset.
    This is critical for recommendation evaluation where rating distributions are often
    highly skewed (e.g., users give more high ratings than low ratings).
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
        """
        Initialize stratified K-fold cross-validator.

        Args:
            n_splits: Number of folds (typically 3-10)
            random_state: Random seed for reproducible splits
            shuffle: Whether to shuffle data before splitting
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split_dataset(self, dataset: Dataset) -> Generator[tuple[Dataset, Dataset], None, None]:
        """
        Generate stratified train/validation dataset pairs for K-fold CV.

        Args:
            dataset: Full dataset to split

        Yields:
            Tuple of (train_dataset, val_dataset) for each fold
        """
        ratings = dataset.ratings.copy()

        # Create rating bins for stratification
        ratings["rating_bin"] = pd.cut(
            ratings["my_rating"],
            bins=[0, 3, 5, 7, 8, 10],
            labels=["low", "below_avg", "avg", "good", "excellent"],
            include_lowest=True,
        )

        # Initialize stratified K-fold splitter
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        # Generate splits
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ratings, ratings["rating_bin"])):
            # Create fold datasets
            train_ratings = (
                ratings.iloc[train_idx].drop("rating_bin", axis=1).reset_index(drop=True)
            )
            val_ratings = ratings.iloc[val_idx].drop("rating_bin", axis=1).reset_index(drop=True)

            train_dataset = Dataset(ratings=train_ratings, watchlist=dataset.watchlist)
            val_dataset = Dataset(ratings=val_ratings, watchlist=dataset.watchlist)

            print(
                f"📂 Fold {fold_idx + 1}/{self.n_splits}: "
                f"train={len(train_ratings)}, val={len(val_ratings)}"
            )

            yield train_dataset, val_dataset


class TemporalKFoldCV:
    """
    Temporal K-fold cross-validation for recommendation systems.

    Respects chronological order by splitting data temporally. Earlier folds
    use older data for training and newer data for validation. This simulates
    realistic deployment where models are trained on historical data and
    evaluated on future data.
    """

    def __init__(self, n_splits: int = 5):
        """
        Initialize temporal K-fold cross-validator.

        Args:
            n_splits: Number of temporal splits (typically 3-7)
        """
        self.n_splits = n_splits

    def split_dataset(self, dataset: Dataset) -> Generator[tuple[Dataset, Dataset], None, None]:
        """
        Generate temporal train/validation dataset pairs for K-fold CV.

        Args:
            dataset: Full dataset to split (must have 'date_rated' column)

        Yields:
            Tuple of (train_dataset, val_dataset) for each fold
        """
        ratings = dataset.ratings.copy()

        # Check for temporal column
        if "date_rated" not in ratings.columns:
            raise ValueError("Temporal K-fold requires 'date_rated' column in ratings")

        # Convert to datetime and sort
        if not pd.api.types.is_datetime64_any_dtype(ratings["date_rated"]):
            ratings["date_rated"] = pd.to_datetime(ratings["date_rated"])

        ratings = ratings.sort_values("date_rated").reset_index(drop=True)

        # Use TimeSeriesSplit for temporal splitting
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(ratings)):
            # Create fold datasets
            train_ratings = ratings.iloc[train_idx].reset_index(drop=True)
            val_ratings = ratings.iloc[val_idx].reset_index(drop=True)

            train_dataset = Dataset(ratings=train_ratings, watchlist=dataset.watchlist)
            val_dataset = Dataset(ratings=val_ratings, watchlist=dataset.watchlist)

            # Show temporal ranges
            train_start = train_ratings["date_rated"].min()
            train_end = train_ratings["date_rated"].max()
            val_start = val_ratings["date_rated"].min()
            val_end = val_ratings["date_rated"].max()

            print(f"📅 Fold {fold_idx + 1}/{self.n_splits}:")
            print(
                f"   Train: {train_start.date()} to {train_end.date()} "
                f"({len(train_ratings)} ratings)"
            )
            print(f"   Val:   {val_start.date()} to {val_end.date()} ({len(val_ratings)} ratings)")

            yield train_dataset, val_dataset


class CrossValidationEvaluator:
    """
    Main evaluation engine for K-fold cross-validation of recommender systems.

    Handles both stratified and temporal cross-validation, evaluates models
    on multiple metrics, and aggregates results across folds with proper
    statistical reporting.
    """

    def __init__(self, cv_strategy: str = "stratified", n_splits: int = 5, random_state: int = 42):
        """
        Initialize cross-validation evaluator.

        Args:
            cv_strategy: "stratified" or "temporal"
            n_splits: Number of CV folds
            random_state: Random seed for reproducible results
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state

        # Initialize CV splitter
        if cv_strategy == "stratified":
            self.cv_splitter = StratifiedKFoldCV(n_splits, random_state)
        elif cv_strategy == "temporal":
            self.cv_splitter = TemporalKFoldCV(n_splits)
        else:
            raise ValueError("cv_strategy must be 'stratified' or 'temporal'")

    def evaluate_recommender(
        self,
        recommender_class: type[RecommenderAlgo],
        dataset: Dataset,
        recommender_params: dict[str, Any] | None = None,
        score_params: dict[str, Any] | None = None,
    ) -> CrossValidationResult:
        """
        Perform K-fold cross-validation evaluation of a recommender.

        Args:
            recommender_class: Class of recommender to evaluate
            dataset: Full dataset for cross-validation
            recommender_params: Parameters for recommender initialization
            score_params: Parameters for scoring/recommendation generation

        Returns:
            CrossValidationResult with aggregated metrics across folds
        """
        print(f"🔄 Starting {self.n_splits}-fold {self.cv_strategy} cross-validation...")
        print(f"   Recommender: {recommender_class.__name__}")
        print(
            f"   Dataset: {len(dataset.ratings)} ratings, {len(dataset.watchlist)} watchlist items"
        )

        # Default parameters
        recommender_params = recommender_params or {}
        score_params = score_params or {"user_weight": 0.7, "global_weight": 0.3}

        fold_results = []

        # Evaluate each fold
        for fold_idx, (train_dataset, val_dataset) in enumerate(
            self.cv_splitter.split_dataset(dataset)
        ):
            print(f"\n🔍 Evaluating Fold {fold_idx + 1}/{self.n_splits}...")

            try:
                # Initialize recommender with training data
                recommender_params["random_seed"] = self.random_state + fold_idx
                recommender = recommender_class(train_dataset, **recommender_params)

                # Fit the model (if supported)
                if hasattr(recommender, "fit"):
                    print(f"   🎓 Training {recommender_class.__name__}...")
                    recommender.fit(**score_params)

                # Get validation items
                val_items = val_dataset.ratings["imdb_const"].tolist()

                # Generate recommendations
                print(f"   🎯 Generating recommendations for {len(val_items)} validation items...")
                scores, _ = recommender.score(
                    seeds=[],
                    exclude_rated=False,  # Don't exclude rated items for validation
                    **score_params,
                )

                # Filter scores to validation items only
                val_scores = {k: v for k, v in scores.items() if k in val_items}

                print(
                    f"   📊 Generated scores for {len(val_scores)}/"
                    f"{len(val_items)} validation items"
                )

                # Calculate metrics
                fold_result = self._evaluate_fold(fold_idx, train_dataset, val_dataset, val_scores)
                fold_results.append(fold_result)

                print(
                    f"   📊 Fold {fold_idx + 1} Results: RMSE={fold_result.rmse:.3f}, "
                    f"P@10={fold_result.precision_at_10:.3f}, R@10={fold_result.recall_at_10:.3f}"
                )

            except Exception as e:
                print(f"   ❌ Fold {fold_idx + 1} failed: {str(e)}")
                # Create dummy result to maintain fold count
                fold_results.append(
                    CVFoldResult(
                        fold_idx=fold_idx,
                        train_size=len(train_dataset.ratings),
                        val_size=len(val_dataset.ratings),
                        rmse=np.inf,
                        mae=np.inf,
                        r2=-np.inf,
                        precision_at_10=0.0,
                        recall_at_10=0.0,
                        ndcg_at_10=0.0,
                    )
                )

        # Aggregate results across folds
        print(f"\n📈 Aggregating results across {len(fold_results)} folds...")
        return self._aggregate_results(fold_results)

    def _evaluate_fold(
        self, fold_idx: int, train_dataset: Dataset, val_dataset: Dataset, scores: dict[str, float]
    ) -> CVFoldResult:
        """Evaluate a single fold and return metrics."""
        val_ratings = val_dataset.ratings

        # Rating prediction metrics
        y_true, y_pred = [], []
        for _, row in val_ratings.iterrows():
            const = row["imdb_const"]
            if const in scores:
                y_true.append(row["my_rating"])
                # Convert recommendation score to rating scale (1-10)
                pred_rating = min(10, max(1, scores[const] * 3 + 5.5))
                y_pred.append(pred_rating)

        if len(y_true) > 0:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        else:
            rmse, mae, r2 = np.inf, np.inf, -np.inf

        # Ranking metrics
        precision_at_10, recall_at_10, ndcg_at_10 = self._calculate_ranking_metrics(
            scores, val_ratings
        )

        return CVFoldResult(
            fold_idx=fold_idx,
            train_size=len(train_dataset.ratings),
            val_size=len(val_ratings),
            rmse=rmse,
            mae=mae,
            r2=r2,
            precision_at_10=precision_at_10,
            recall_at_10=recall_at_10,
            ndcg_at_10=ndcg_at_10,
        )

    def _calculate_ranking_metrics(
        self, scores: dict[str, float], val_ratings: pd.DataFrame
    ) -> tuple[float, float, float]:
        """Calculate precision@10, recall@10, and NDCG@10."""
        if not scores:
            return 0.0, 0.0, 0.0

        # Ground truth: items rated >= 8
        relevant_items = set(val_ratings[val_ratings["my_rating"] >= 8]["imdb_const"].tolist())

        if not relevant_items:
            return 0.0, 0.0, 0.0

        # Top-10 recommendations
        top_10_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        recommended_items = {item[0] for item in top_10_items}

        # Precision@10
        hits = len(recommended_items & relevant_items)
        precision_at_10 = hits / min(10, len(recommended_items)) if recommended_items else 0.0

        # Recall@10
        recall_at_10 = hits / len(relevant_items) if relevant_items else 0.0

        # NDCG@10 (simplified version)
        dcg = 0.0
        for i, (item, _score) in enumerate(top_10_items):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

        # Ideal DCG (assuming all relevant items are ranked first)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(relevant_items))))

        ndcg_at_10 = dcg / idcg if idcg > 0 else 0.0

        return precision_at_10, recall_at_10, ndcg_at_10

    def _aggregate_results(self, fold_results: list[CVFoldResult]) -> CrossValidationResult:
        """Aggregate results across all folds with statistics."""
        # Filter out failed folds (infinite RMSE)
        valid_results = [r for r in fold_results if np.isfinite(r.rmse)]

        if not valid_results:
            print("⚠️  No valid fold results found!")
            # Return dummy results
            return CrossValidationResult(
                n_folds=len(fold_results),
                fold_results=fold_results,
                mean_rmse=np.inf,
                std_rmse=0.0,
                mean_mae=np.inf,
                std_mae=0.0,
                mean_r2=-np.inf,
                std_r2=0.0,
                mean_precision_at_10=0.0,
                std_precision_at_10=0.0,
                mean_recall_at_10=0.0,
                std_recall_at_10=0.0,
                mean_ndcg_at_10=0.0,
                std_ndcg_at_10=0.0,
            )

        # Calculate means and standard deviations
        rmse_values = [r.rmse for r in valid_results]
        mae_values = [r.mae for r in valid_results]
        r2_values = [r.r2 for r in valid_results]
        precision_values = [r.precision_at_10 for r in valid_results]
        recall_values = [r.recall_at_10 for r in valid_results]
        ndcg_values = [r.ndcg_at_10 for r in valid_results]

        result = CrossValidationResult(
            n_folds=len(fold_results),
            fold_results=fold_results,
            mean_rmse=np.mean(rmse_values),
            std_rmse=np.std(rmse_values, ddof=1) if len(rmse_values) > 1 else 0.0,
            mean_mae=np.mean(mae_values),
            std_mae=np.std(mae_values, ddof=1) if len(mae_values) > 1 else 0.0,
            mean_r2=np.mean(r2_values),
            std_r2=np.std(r2_values, ddof=1) if len(r2_values) > 1 else 0.0,
            mean_precision_at_10=np.mean(precision_values),
            std_precision_at_10=(
                np.std(precision_values, ddof=1) if len(precision_values) > 1 else 0.0
            ),
            mean_recall_at_10=np.mean(recall_values),
            std_recall_at_10=np.std(recall_values, ddof=1) if len(recall_values) > 1 else 0.0,
            mean_ndcg_at_10=np.mean(ndcg_values),
            std_ndcg_at_10=np.std(ndcg_values, ddof=1) if len(ndcg_values) > 1 else 0.0,
        )

        # Print summary
        print(f"\n📈 {self.cv_strategy.title()} {self.n_splits}-Fold Cross-Validation Results:")
        print(f"   Valid folds: {len(valid_results)}/{len(fold_results)}")
        print(f"   RMSE: {result.mean_rmse:.3f} ± {result.std_rmse:.3f}")
        print(f"   MAE:  {result.mean_mae:.3f} ± {result.std_mae:.3f}")
        print(f"   R²:   {result.mean_r2:.3f} ± {result.std_r2:.3f}")
        print(f"   P@10: {result.mean_precision_at_10:.3f} ± {result.std_precision_at_10:.3f}")
        print(f"   R@10: {result.mean_recall_at_10:.3f} ± {result.std_recall_at_10:.3f}")
        print(f"   NDCG@10: {result.mean_ndcg_at_10:.3f} ± {result.std_ndcg_at_10:.3f}")

        return result


def compare_models_cv(
    dataset: Dataset,
    recommender_configs: dict[str, dict[str, Any]],
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, CrossValidationResult]:
    """
    Compare multiple recommender models using K-fold cross-validation.

    Args:
        dataset: Dataset for evaluation
        recommender_configs: Dict of {model_name: {class, params, score_params}}
        cv_strategy: "stratified" or "temporal"
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dict of {model_name: CrossValidationResult}
    """
    print("🏆 Model Comparison via K-Fold Cross-Validation")
    print("=" * 60)

    evaluator = CrossValidationEvaluator(cv_strategy, n_splits, random_state)
    results = {}

    for model_name, config in recommender_configs.items():
        print(f"\n🎯 Evaluating {model_name}...")

        try:
            result = evaluator.evaluate_recommender(
                recommender_class=config["class"],
                dataset=dataset,
                recommender_params=config.get("params", {}),
                score_params=config.get("score_params", {}),
            )
            results[model_name] = result

        except Exception as e:
            print(f"❌ {model_name} evaluation failed: {str(e)}")
            continue

    # Print comparison summary
    print(f"\n🏆 Model Comparison Summary ({cv_strategy} {n_splits}-fold CV):")
    print("-" * 80)
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'P@10':<12} {'R@10':<12} {'NDCG@10':<12}")
    print("-" * 80)

    for model_name, result in results.items():
        rmse_str = f"{result.mean_rmse:.3f}±{result.std_rmse:.3f}"
        mae_str = f"{result.mean_mae:.3f}±{result.std_mae:.3f}"
        p10_str = f"{result.mean_precision_at_10:.3f}±{result.std_precision_at_10:.3f}"
        r10_str = f"{result.mean_recall_at_10:.3f}±{result.std_recall_at_10:.3f}"
        ndcg_str = f"{result.mean_ndcg_at_10:.3f}±{result.std_ndcg_at_10:.3f}"

        print(
            f"{model_name:<20} "
            f"{rmse_str:<12} "
            f"{mae_str:<12} "
            f"{p10_str:<12} "
            f"{r10_str:<12} "
            f"{ndcg_str:<12}"
        )

    return results
