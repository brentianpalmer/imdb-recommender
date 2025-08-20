"""
Scikit-learn Integration Module

Provides sklearn-compatible wrapper for the SVD recommender to leverage
cross_validate, GridSearchCV, and other sklearn utilities.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection._split import _BaseKFold

from .cross_validation import StratifiedKFoldCV, TemporalKFoldCV
from .data_io import Dataset
from .recommender_base import RecommenderAlgo
from .recommender_svd import SVDAutoRecommender

warnings.filterwarnings("ignore", category=UserWarning)


class RecommenderSplitter(_BaseKFold):
    """Custom CV splitter for recommender systems that works with sklearn."""

    def __init__(self, n_splits=5, strategy="stratified", random_state=None):
        super().__init__(n_splits, shuffle=False, random_state=random_state)
        self.strategy = strategy
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        """Generate indices for test sets."""
        if hasattr(X, "dataset"):
            dataset = X.dataset
        else:
            # Fallback - use simple random splitting
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            np.random.RandomState(self.random_state).shuffle(indices)
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[: n_samples % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                yield indices[start:stop]
                current = stop
            return

        # Use custom splitting logic for datasets
        if self.strategy == "temporal":
            cv = TemporalKFoldCV(n_splits=self.n_splits, random_state=self.random_state)
        else:  # stratified
            cv = StratifiedKFoldCV(n_splits=self.n_splits, random_state=self.random_state)

        # Get train/test splits and yield test indices
        splits = list(cv.split(dataset))
        for _, test_idx in splits:
            yield test_idx


class BaseRecommenderEstimator(BaseEstimator, RegressorMixin):
    """Base class for sklearn-compatible recommender estimators."""

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
        self.dataset = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> BaseRecommenderEstimator:
        """Fit the recommender model."""
        # Store dataset for later use
        if hasattr(X, "dataset"):
            self.dataset = X.dataset
        else:
            raise ValueError("X must have a 'dataset' attribute")
        return self

    def _create_recommender(self, dataset: Dataset) -> RecommenderAlgo:
        """Create the specific recommender instance. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _create_recommender")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions. Override in subclasses for algorithm-specific logic."""
        raise NotImplementedError("Subclasses must implement predict")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return RMSE score (negated for sklearn compatibility)."""
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return -rmse  # Negative because sklearn maximizes scores


class SVDEstimator(BaseRecommenderEstimator):
    """Sklearn-compatible wrapper for SVDAutoRecommender with full hyperparameter support."""

    def __init__(
        self,
        user_weight: float = 0.7,
        global_weight: float = 0.3,
        recency: float = 0.1,
        random_seed: int = 42,
        n_factors: int = 24,
        reg_param: float = 0.05,
        n_iter: int = 20,
    ):
        super().__init__(user_weight, global_weight, recency, random_seed)
        self.n_factors = n_factors
        self.reg_param = reg_param
        self.n_iter = n_iter

    def _create_recommender(self, dataset: Dataset) -> RecommenderAlgo:
        recommender = SVDAutoRecommender(dataset, random_seed=self.random_seed)

        # Apply hyperparameters
        recommender.apply_hyperparameters(
            {
                "n_factors": self.n_factors,
                "reg_param": self.reg_param,
                "n_iter": self.n_iter,
            }
        )
        return recommender

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using SVD."""
        # Create and configure the recommender
        recommender = self._create_recommender(self.dataset)

        predictions = []

        for row in X:
            _, item_id = int(row[0]), int(row[1])  # user_id not used in SVD scoring
            seeds = [item_id] if item_id else []

            # Generate scores
            scores, _ = recommender.score(
                seeds, self.user_weight, self.global_weight, self.recency, exclude_rated=False
            )

            # Extract prediction score for the specific item_id, default to 0 if not found
            prediction = scores.get(str(item_id), 0.0)
            predictions.append(prediction)

        return np.array(predictions)


def sklearn_cross_validate(
    estimator: BaseRecommenderEstimator,
    dataset: Dataset,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str | list[str] = "neg_root_mean_squared_error",
    n_jobs: int = 1,
    return_train_score: bool = False,
):
    """Cross-validate a recommender using sklearn with custom dataset handling."""

    # Create a minimal array-like object that carries the dataset
    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.shape = (len(dataset.ratings), 2)  # (user_id, item_id) pairs

        def __len__(self):
            return self.shape[0]

        def __getitem__(self, idx):
            # Simple implementation - return user/item pairs
            ratings = (
                self.dataset.ratings.iloc[idx]
                if hasattr(idx, "__iter__")
                else [self.dataset.ratings.iloc[idx]]
            )
            return np.array([[0, hash(r["imdb_const"]) % 1000] for _, r in ratings.iterrows()])

    # Prepare data
    X = DatasetWrapper(dataset)
    y = dataset.ratings["my_rating"].values

    # Use custom splitter
    cv = RecommenderSplitter(n_splits=n_splits, strategy=cv_strategy, random_state=random_state)

    # Run cross-validation
    return cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        error_score="raise",
    )


def sklearn_grid_search(
    estimator: BaseRecommenderEstimator,
    dataset: Dataset,
    param_grid: dict[str, Any],
    cv_strategy: str = "stratified",
    n_splits: int = 3,
    random_state: int = 42,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = 1,
    verbose: int = 1,
):
    """Grid search for recommender hyperparameters using sklearn."""

    # Create dataset wrapper
    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.shape = (len(dataset.ratings), 2)

        def __len__(self):
            return self.shape[0]

        def __getitem__(self, idx):
            ratings = (
                self.dataset.ratings.iloc[idx]
                if hasattr(idx, "__iter__")
                else [self.dataset.ratings.iloc[idx]]
            )
            return np.array([[0, hash(r["imdb_const"]) % 1000] for _, r in ratings.iterrows()])

    # Prepare data
    X = DatasetWrapper(dataset)
    y = dataset.ratings["my_rating"].values

    # Custom CV splitter
    cv = RecommenderSplitter(n_splits=n_splits, strategy=cv_strategy, random_state=random_state)

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
    )

    # Fit and return
    grid_search.fit(X, y)
    return grid_search


def compare_models_sklearn(
    dataset: Dataset,
    estimators: dict[str, BaseRecommenderEstimator] = None,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str | list[str] = "neg_root_mean_squared_error",
):
    """Compare multiple recommender models using sklearn cross-validation."""

    if estimators is None:
        # Default SVD comparison
        estimators = {
            "SVD_Default": SVDEstimator(random_seed=random_state),
            "SVD_Optimal": SVDEstimator(
                n_factors=24, reg_param=0.05, n_iter=20, random_seed=random_state
            ),
        }

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
