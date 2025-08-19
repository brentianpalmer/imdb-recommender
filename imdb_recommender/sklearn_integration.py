"""
Scikit-learn Integration Module

Provides sklearn-compatible wrappers for IMDb recommenders to leverage
cross_validate, GridSearchCV, and other sklearn utilities.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection._split import _BaseKFold

from .cross_validation import StratifiedKFoldCV, TemporalKFoldCV
from .data_io import Dataset
from .recommender_all_in_one import AllInOneRecommender
from .recommender_base import RecommenderAlgo
from .recommender_pop import PopSimRecommender
from .recommender_svd import SVDAutoRecommender

warnings.filterwarnings("ignore", category=UserWarning)


class RecommenderSplitter(_BaseKFold):
    """Custom CV splitter for recommender systems that works with sklearn."""

    def __init__(self, cv_strategy: str = "stratified", n_splits: int = 5, random_state: int = 42):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self._dataset = None

    def _iter_test_masks(self, X, y=None, groups=None):
        """Generate boolean masks for each fold."""
        if self._dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")

        if self.cv_strategy == "stratified":
            cv_splitter = StratifiedKFoldCV(n_splits=self.n_splits, random_state=self.random_state)
        elif self.cv_strategy == "temporal":
            cv_splitter = TemporalKFoldCV(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown cv_strategy: {self.cv_strategy}")

        # Generate train/val datasets from our custom splitter
        fold_datasets = list(cv_splitter.split_dataset(self._dataset))

        for _train_dataset, val_dataset in fold_datasets:
            # Create boolean mask for validation indices
            val_indices = val_dataset.ratings.index.tolist()
            test_mask = np.zeros(len(self._dataset.ratings), dtype=bool)
            test_mask[val_indices] = True
            yield test_mask

    def set_dataset(self, dataset: Dataset):
        """Set the dataset for splitting."""
        self._dataset = dataset

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits


class BaseRecommenderEstimator(BaseEstimator, RegressorMixin):
    """Base sklearn-compatible wrapper for IMDb recommenders."""

    def __init__(
        self,
        user_weight: float = 0.7,
        global_weight: float = 0.3,
        recency: float = 0.1,
        random_seed: int = 42,
    ):
        self.user_weight = user_weight
        self.global_weight = global_weight
        self.recency = recency
        self.random_seed = random_seed
        self._recommender = None
        self._dataset = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the recommender on training data."""
        # X should be ratings DataFrame, y is ignored (unsupervised)
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with ratings data")

        # Create empty watchlist with required columns
        empty_watchlist = pd.DataFrame(columns=["imdb_const"])

        # Create dataset from ratings
        self._dataset = Dataset(ratings=X, watchlist=empty_watchlist)

        # Initialize the recommender (subclasses implement _create_recommender)
        self._recommender = self._create_recommender(self._dataset)

        # Fit if the recommender supports it
        if hasattr(self._recommender, "fit"):
            self._recommender.fit(
                user_weight=self.user_weight, global_weight=self.global_weight, recency=self.recency
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ratings for given movie items."""
        if self._recommender is None:
            raise ValueError("Model must be fitted before making predictions")

        # X contains movie constants to predict ratings for
        if "imdb_const" in X.columns:
            items = X["imdb_const"].tolist()
        else:
            items = X.index.tolist()

        # Generate scores
        scores, _ = self._recommender.score(
            seeds=[],
            exclude_rated=False,
            user_weight=self.user_weight,
            global_weight=self.global_weight,
            recency=self.recency,
        )

        # Convert scores to rating predictions (1-10 scale)
        predictions = []
        for item in items:
            if item in scores:
                # Convert recommendation score to rating scale
                pred_rating = min(10, max(1, scores[item] * 3 + 5.5))
                predictions.append(pred_rating)
            else:
                # Default prediction for unseen items
                predictions.append(5.5)

        return np.array(predictions)

    def score(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> float:
        """Return the negative RMSE (higher is better for sklearn)."""
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return -rmse  # Negative because sklearn maximizes score

    def _create_recommender(self, dataset: Dataset) -> RecommenderAlgo:
        """Create the specific recommender instance (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _create_recommender")


class PopSimEstimator(BaseRecommenderEstimator):
    """Sklearn-compatible wrapper for PopSimRecommender."""

    def _create_recommender(self, dataset: Dataset) -> RecommenderAlgo:
        return PopSimRecommender(dataset, random_seed=self.random_seed)


class SVDEstimator(BaseRecommenderEstimator):
    """Sklearn-compatible wrapper for SVDAutoRecommender."""

    def _create_recommender(self, dataset: Dataset) -> RecommenderAlgo:
        return SVDAutoRecommender(dataset, random_seed=self.random_seed)


class AllInOneEstimator(BaseRecommenderEstimator):
    """Sklearn-compatible wrapper for AllInOneRecommender."""

    def __init__(
        self,
        user_weight: float = 0.7,
        global_weight: float = 0.3,
        recency: float = 0.1,
        random_seed: int = 42,
        svd_components: int = 50,
        mmr_lambda: float = 0.5,
        min_votes_threshold: int = 100,
        exposure_model_params: dict | None = None,
        preference_model_params: dict | None = None,
    ):
        super().__init__(user_weight, global_weight, recency, random_seed)
        self.svd_components = svd_components
        self.mmr_lambda = mmr_lambda
        self.min_votes_threshold = min_votes_threshold
        self.exposure_model_params = exposure_model_params or {}
        self.preference_model_params = preference_model_params or {}

    def _create_recommender(self, dataset: Dataset) -> RecommenderAlgo:
        recommender = AllInOneRecommender(dataset, random_seed=self.random_seed)

        # Configure hyperparameters
        recommender.svd_components = self.svd_components
        recommender.mmr_lambda = self.mmr_lambda
        recommender.min_votes_threshold = self.min_votes_threshold
        recommender.exposure_model_params = self.exposure_model_params
        recommender.preference_model_params = self.preference_model_params

        return recommender


def sklearn_cross_validate(
    estimator: BaseRecommenderEstimator,
    dataset: Dataset,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str | list[str] = "neg_root_mean_squared_error",
    n_jobs: int = 1,
    return_train_score: bool = False,
) -> dict[str, np.ndarray]:
    """
    Perform cross-validation using sklearn's cross_validate with custom splitter.

    Args:
        estimator: The recommender estimator
        dataset: Full dataset for cross-validation
        cv_strategy: "stratified" or "temporal"
        n_splits: Number of CV folds
        random_state: Random seed
        scoring: Scoring metrics (sklearn format)
        n_jobs: Number of parallel jobs
        return_train_score: Whether to return training scores

    Returns:
        Dictionary with cross-validation results
    """
    # Prepare data in sklearn format
    X = dataset.ratings.copy()
    y = dataset.ratings["my_rating"].copy()

    # Create custom splitter
    cv_splitter = RecommenderSplitter(
        cv_strategy=cv_strategy, n_splits=n_splits, random_state=random_state
    )
    cv_splitter.set_dataset(dataset)

    # Use sklearn's cross_validate
    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        error_score="raise",
    )

    return results


def sklearn_grid_search(
    estimator: BaseRecommenderEstimator,
    dataset: Dataset,
    param_grid: dict[str, Any],
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = 1,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Perform grid search using sklearn's GridSearchCV with custom splitter.

    Args:
        estimator: The recommender estimator
        dataset: Full dataset for hyperparameter tuning
        param_grid: Parameter grid to search
        cv_strategy: "stratified" or "temporal"
        n_splits: Number of CV folds
        random_state: Random seed
        scoring: Scoring metric (sklearn format)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level

    Returns:
        Fitted GridSearchCV object
    """
    # Prepare data in sklearn format
    X = dataset.ratings.copy()
    y = dataset.ratings["my_rating"].copy()

    # Create custom splitter
    cv_splitter = RecommenderSplitter(
        cv_strategy=cv_strategy, n_splits=n_splits, random_state=random_state
    )
    cv_splitter.set_dataset(dataset)

    # Create and fit GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
    )

    grid_search.fit(X, y)

    return grid_search


# Convenience functions for backward compatibility
def compare_models_sklearn(
    dataset: Dataset,
    estimators: dict[str, BaseRecommenderEstimator],
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    scoring: list[str] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compare multiple models using sklearn cross-validation.

    Args:
        dataset: Dataset to evaluate on
        estimators: Dict mapping model names to estimator instances
        cv_strategy: Cross-validation strategy
        n_splits: Number of folds
        random_state: Random seed
        scoring: List of scoring metrics

    Returns:
        Dictionary with results for each model
    """
    if scoring is None:
        scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

    results = {}

    for name, estimator in estimators.items():
        print(f"\n📊 Evaluating {name}...")

        cv_results = sklearn_cross_validate(
            estimator=estimator,
            dataset=dataset,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            random_state=random_state,
            scoring=scoring,
            n_jobs=1,  # Keep sequential for now due to custom dataset handling
        )

        results[name] = cv_results

        # Print summary
        test_rmse = -cv_results["test_neg_root_mean_squared_error"]
        print(f"   RMSE: {np.mean(test_rmse):.3f} ± {np.std(test_rmse):.3f}")

        if "test_neg_mean_absolute_error" in cv_results:
            test_mae = -cv_results["test_neg_mean_absolute_error"]
            print(f"   MAE:  {np.mean(test_mae):.3f} ± {np.std(test_mae):.3f}")

        if "test_r2" in cv_results:
            test_r2 = cv_results["test_r2"]
            print(f"   R²:   {np.mean(test_r2):.3f} ± {np.std(test_r2):.3f}")

    return results
