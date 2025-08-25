"""ElasticNet-based movie recommender."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MultiLabelBinarizer

from .recommender_base import RecommenderAlgo
from .data_io import Dataset


@dataclass
class ElasticNetRecommendation:
    """Single recommendation from ElasticNet model."""

    imdb_const: str
    title: str | None
    year: int | None
    genres: str | None
    score: float
    why_explainer: str


class ElasticNetRecommender(RecommenderAlgo):
    """ElasticNet-based movie recommender with feature engineering."""

    def __init__(
        self,
        dataset: Dataset,
        alpha: float = 0.1,
        l1_ratio: float = 0.1,
        random_seed: int = 42,
        top_directors: int = 30,
    ):
        """Initialize ElasticNet recommender.

        Parameters
        ----------
        dataset : Dataset
            The training dataset with ratings and watchlist
        alpha : float
            ElasticNet regularization strength
        l1_ratio : float
            ElasticNet L1 ratio (0=Ridge, 1=Lasso)
        random_seed : int
            Random state for reproducibility
        top_directors : int
            Number of top directors to include as features
        """
        super().__init__(dataset, random_seed)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.top_directors = top_directors
        self.model = None
        self.feature_columns = None
        self.numeric_columns = None
        self.training_stats = None

    def _safe_numeric(self, s):
        """Convert to numeric, handling errors."""
        return pd.to_numeric(s, errors="coerce")

    def _split_listfield(self, s):
        """Split comma-separated string into list."""
        if pd.isna(s):
            return []
        return [t.strip() for t in str(s).split(",") if t and str(t).strip()]

    def _engineer_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Engineer features for ElasticNet model."""
        # Map common column names to standard names
        rename_map = {
            "Const": "imdb_const",
            "Your Rating": "my_rating",
            "Title Type": "title_type",
            "Runtime (mins)": "runtime",
            "IMDb Rating": "imdb_rating",
            "Num Votes": "num_votes",
            "Date Rated": "rated_at",
            "Release Date": "release_date",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Target variable (only for training data)
        y = None
        if "my_rating" in df.columns:
            y = self._safe_numeric(df["my_rating"]).astype(float).values

        # Initialize feature dataframe
        features = pd.DataFrame(index=df.index)

        # Numeric features with missing value handling
        numeric_cols = ["year", "runtime", "imdb_rating", "num_votes"]
        for col in numeric_cols:
            if col in df.columns:
                features[col] = self._safe_numeric(df[col])
                if col == "imdb_rating" and features[col].isna().any():
                    # Use global median for missing IMDB ratings
                    median_rating = features[col].median() if features[col].notna().any() else 7.0
                    features[col] = features[col].fillna(median_rating)
                else:
                    features[col] = features[col].fillna(features[col].median())
            else:
                # Set default values for missing columns
                if col == "imdb_rating":
                    features[col] = 7.0
                elif col == "num_votes":
                    features[col] = 1000.0
                elif col == "year":
                    features[col] = 2000.0
                else:
                    features[col] = 0.0

        # Cap extreme values
        features["num_votes"] = features["num_votes"].clip(upper=2_000_000)
        features["log_votes"] = np.log1p(features["num_votes"])
        features["year"] = features["year"].clip(lower=1900, upper=2024)  # Exclude future years
        features["decade"] = (features["year"] // 10) * 10

        # Date features
        if "rated_at" in df.columns and df["rated_at"].notna().any():
            features["rated_at_dt"] = pd.to_datetime(df["rated_at"], errors="coerce")
            first_rating = features["rated_at_dt"].min()
            if pd.notna(first_rating):
                features["days_since_first"] = (
                    features["rated_at_dt"] - first_rating
                ).dt.days.fillna(0)
                features["rate_year"] = features["rated_at_dt"].dt.year.fillna(2020).astype(int)
                features["rate_month"] = features["rated_at_dt"].dt.month.fillna(6).astype(int)
                features["rate_dow"] = features["rated_at_dt"].dt.dayofweek.fillna(3).astype(int)
            else:
                features["days_since_first"] = 0.0
                features["rate_year"] = 2020
                features["rate_month"] = 6
                features["rate_dow"] = 3
        else:
            features["days_since_first"] = 0.0
            features["rate_year"] = 2020
            features["rate_month"] = 6
            features["rate_dow"] = 3

        # Release date features
        if "release_date" in df.columns:
            features["release_dt"] = pd.to_datetime(df["release_date"], errors="coerce")
            ref_date = pd.Timestamp.now()
            age = (ref_date - features["release_dt"]).dt.days
            features["age_days"] = age.fillna(age.median() if age.notna().any() else 3650)
            features["release_month"] = features["release_dt"].dt.month.fillna(6).astype(int)
        else:
            features["age_days"] = 3650.0  # ~10 years default
            features["release_month"] = 6

        # One-hot encode categorical features
        if "title_type" in df.columns:
            title_type_dummies = pd.get_dummies(df["title_type"].fillna("Movie"), prefix="type")
            features = pd.concat([features, title_type_dummies], axis=1)

        decade_dummies = pd.get_dummies(features["decade"], prefix="decade")
        features = pd.concat([features, decade_dummies], axis=1)

        month_dummies = pd.get_dummies(features["rate_month"], prefix="rate_month")
        features = pd.concat([features, month_dummies], axis=1)

        # Genre features (multi-hot encoding)
        if "genres" in df.columns:
            genre_lists = (
                df["genres"]
                .fillna("")
                .apply(lambda s: [g.strip() for g in str(s).split(",") if g.strip()])
            )
            mlb = MultiLabelBinarizer()
            genre_encoded = mlb.fit_transform(genre_lists)
            genre_df = pd.DataFrame(
                genre_encoded, columns=[f"genre_{g}" for g in mlb.classes_], index=df.index
            )
            features = pd.concat([features, genre_df], axis=1)

        # Director features (top-K one-hot)
        if "directors" in df.columns:
            director_lists = df["directors"].apply(self._split_listfield)
            director_counts = Counter([d for lst in director_lists for d in lst if d])
            top_dirs = {d for d, _ in director_counts.most_common(self.top_directors)}

            def normalize_name(name):
                return re.sub(r"[^A-Za-z0-9]+", "_", str(name))[:30]

            for director in top_dirs:
                col_name = f"dir_{normalize_name(director)}"
                features[col_name] = director_lists.apply(
                    lambda lst: 1.0 if director in lst else 0.0
                )

        # Store numeric column names for standardization
        self.numeric_columns = [
            "year",
            "runtime",
            "imdb_rating",
            "log_votes",
            "days_since_first",
            "age_days",
        ]

        # Convert any remaining datetime columns to numeric
        datetime_cols = features.select_dtypes(include=["datetime"]).columns
        for col in datetime_cols:
            if col == "rated_at_dt":
                # Convert to unix timestamp (nanoseconds to seconds)
                features[col] = features[col].astype("int64") // 10**9
            elif col == "release_dt":
                # Convert to unix timestamp (nanoseconds to seconds)
                features[col] = features[col].astype("int64") // 10**9
            else:
                # Generic datetime to unix timestamp conversion
                features[col] = features[col].astype("int64") // 10**9

        # Fill any remaining NaNs with 0
        features = features.fillna(0.0)

        return features, y

    def _standardize_features(
        self, X_train: pd.DataFrame, X_pred: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Standardize numeric features using training statistics."""
        X_train = X_train.copy()
        X_pred = X_pred.copy()

        # Calculate stats from training data
        numeric_cols = [col for col in self.numeric_columns if col in X_train.columns]
        means = X_train[numeric_cols].mean()
        stds = X_train[numeric_cols].std().replace(0, 1.0)

        # Standardize both datasets
        X_train[numeric_cols] = (X_train[numeric_cols] - means) / stds
        X_pred[numeric_cols] = (X_pred[numeric_cols] - means) / stds

        return X_train, X_pred

    def _filter_unreleased_and_insufficient_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out movies with future release dates or insufficient metadata.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with movie data

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame excluding unreleased/insufficient content
        """
        import datetime

        current_date = datetime.datetime.now()
        current_year = current_date.year

        # Create filter conditions
        conditions = []

        # 1. Exclude future years (movies not yet released)
        if "year" in df.columns:
            year_condition = df["year"].isna() | (
                pd.to_numeric(df["year"], errors="coerce") <= current_year
            )
            conditions.append(year_condition)

        # 2. Exclude movies with future release dates
        if "release_date" in df.columns:
            release_dates = pd.to_datetime(df["release_date"], errors="coerce")
            release_condition = release_dates.isna() | (release_dates <= current_date)
            conditions.append(release_condition)

        # 3. Exclude movies with insufficient metadata (likely unreleased)
        metadata_conditions = []

        # Must have some IMDb rating (not 0 or NaN)
        if "imdb_rating" in df.columns:
            rating_condition = pd.to_numeric(df["imdb_rating"], errors="coerce").fillna(0) > 0
            metadata_conditions.append(rating_condition)

        # Must have some votes (not 0)
        if "num_votes" in df.columns:
            votes_condition = pd.to_numeric(df["num_votes"], errors="coerce").fillna(0) > 0
            metadata_conditions.append(votes_condition)

        # Must have valid genres (not just "nan" or empty)
        if "genres" in df.columns:
            genre_condition = (
                df["genres"].notna()
                & (df["genres"].astype(str).str.strip() != "")
                & (df["genres"].astype(str).str.lower() != "nan")
            )
            metadata_conditions.append(genre_condition)

        # Require at least 2 out of 3 metadata conditions to be met
        if len(metadata_conditions) >= 2:
            metadata_filter = sum(metadata_conditions) >= 2
            conditions.append(metadata_filter)
        elif len(metadata_conditions) == 1:
            conditions.append(metadata_conditions[0])

        # Combine all conditions
        if conditions:
            final_condition = conditions[0]
            for condition in conditions[1:]:
                final_condition = final_condition & condition

            filtered_df = df[final_condition].copy()

            # Log filtering results for debugging
            original_count = len(df)
            filtered_count = len(filtered_df)
            if original_count > filtered_count:
                print(
                    f"🔍 ElasticNet: Filtered out {original_count - filtered_count} "
                    f"unreleased/insufficient items ({filtered_count} remaining)"
                )

            return filtered_df

        return df

    def fit(self) -> None:
        """Train the ElasticNet model on the dataset."""
        # Prepare training data from ratings
        ratings_df = self.dataset.ratings.copy()

        # Engineer features
        X_train, y_train = self._engineer_features(ratings_df)

        if y_train is None:
            raise ValueError("No rating information found for training")

        # Store feature columns and training stats
        self.feature_columns = X_train.columns.tolist()
        self.training_stats = {
            "y_mean": np.mean(y_train),
            "y_std": np.std(y_train),
            "y_min": np.min(y_train),
            "y_max": np.max(y_train),
        }

        # Initialize and train model
        self.model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_seed, max_iter=5000
        )
        self.model.fit(X_train, y_train)

    def _scale_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Scale raw predictions to 1-10 range using training statistics."""
        if self.training_stats is None:
            return np.clip(predictions, 1.0, 10.0)

        # Normalize using training statistics
        y_mean = self.training_stats["y_mean"]
        y_std = self.training_stats["y_std"]

        # Z-score normalize
        pred_normalized = (predictions - y_mean) / y_std

        # Apply bounded transformation and scale to 1-10
        pred_bounded = np.tanh(pred_normalized / 2)
        pred_scaled = 5.5 + 4.5 * pred_bounded

        return np.clip(pred_scaled, 1.0, 10.0)

    def score(
        self,
        seeds: list[str],
        user_weight: float = 0.5,
        global_weight: float = 0.1,
        recency: float = 0.0,
        exclude_rated: bool = True,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Generate recommendations using ElasticNet model.

        Parameters
        ----------
        seeds : list[str]
            Seed movie IDs (not used in ElasticNet, but kept for interface compatibility)
        user_weight : float
            Weight for user preferences (not used in ElasticNet)
        global_weight : float
            Weight for global popularity (not used in ElasticNet)
        recency : float
            Recency bias (not used in ElasticNet)
        exclude_rated : bool
            Whether to exclude already rated movies

        Returns
        -------
        tuple[dict[str, float], dict[str, str]]
            Dictionaries of scores and explanations keyed by movie ID
        """
        if self.model is None:
            self.fit()

        # Get watchlist data for prediction
        watchlist_df = self.dataset.watchlist.copy()

        if len(watchlist_df) == 0:
            return {}, {}

        # Filter out rated movies if requested
        if exclude_rated:
            rated_ids = set(self.dataset.ratings["imdb_const"])
            watchlist_df = watchlist_df[~watchlist_df["imdb_const"].isin(rated_ids)]

        # Filter out unreleased movies and those with insufficient metadata
        watchlist_df = self._filter_unreleased_and_insufficient_metadata(watchlist_df)

        if len(watchlist_df) == 0:
            return {}, {}

        # Engineer features for watchlist
        X_watchlist, _ = self._engineer_features(watchlist_df)

        # Align feature columns with training data
        missing_cols = set(self.feature_columns) - set(X_watchlist.columns)
        extra_cols = set(X_watchlist.columns) - set(self.feature_columns)

        # Add missing columns with zeros
        for col in missing_cols:
            X_watchlist[col] = 0.0

        # Remove extra columns
        X_watchlist = X_watchlist.drop(columns=list(extra_cols))

        # Ensure same column order as training
        X_watchlist = X_watchlist[self.feature_columns]

        # Generate predictions
        raw_predictions = self.model.predict(X_watchlist)
        scaled_predictions = self._scale_predictions(raw_predictions)

        # Create score and explanation dictionaries
        scores = {}
        explanations = {}

        for idx, (_, row) in enumerate(watchlist_df.iterrows()):
            imdb_const = row["imdb_const"]
            score = float(scaled_predictions[idx])
            scores[imdb_const] = score
            explanations[imdb_const] = "ElasticNet feature-engineered prediction"

        return scores, explanations
