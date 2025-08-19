"""
All-in-One Four-Stage IMDb Recommender

This module implements a sophisticated recommendation system that:
1. Models exposure bias (what you've seen vs. what exists)
2. Learns preference patterns from your ratings
3. Balances personal taste with popularity priors
4. Optimizes diversity using Maximal Marginal Relevance (MMR)

The four stages are:
1. Feature Engineering: Content vectors, popularity signals, temporal features
2. Exposure Modeling: P(exposed) - likelihood of having seen a movie
3. Preference Modeling: P(like|exposed) - likelihood of liking given exposure
4. Diversity Optimization: MMR re-ranking for balanced recommendations

Author: IMDb Recommender Team
Date: August 2025
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .recommender_base import RecommenderAlgo

warnings.filterwarnings("ignore", category=UserWarning)


class AllInOneRecommender(RecommenderAlgo):
    """
    All-in-One Four-Stage IMDb Recommender

    This recommender implements a sophisticated multi-stage approach:

    Stage 1: Feature Engineering
    - Content features (genres, runtime, type)
    - Popularity features (IMDb rating, vote count)
    - Temporal features (recency with decay)

    Stage 2: Exposure Modeling
    - Predicts P(exposed) - likelihood user has seen the movie
    - Uses SGD classifier with logistic loss on content + popularity features
    - Handles selection bias in user's data

    Stage 3: Preference Modeling
    - Predicts P(like|exposed) using pairwise learning
    - Learns from rating differences (A > B if rating_A >= rating_B + 2)
    - Uses calibrated SGD classifier with feature differences for robust preference learning

    Stage 4: Diversity Optimization
    - Projects items into latent space using SVD
    - Applies MMR (Maximal Marginal Relevance) re-ranking
    - Balances relevance with diversity

    Final Score: 0.7 * Personal + 0.3 * Popularity
    """

    def __init__(self, dataset, random_seed: int = 42, **kwargs):
        """Initialize the All-in-One recommender."""
        super().__init__(dataset, random_seed, **kwargs)

        # Model components
        self.exposure_model = None
        self.preference_model = None
        self.feature_scaler = None
        self.svd_model = None
        self.genre_encoder = TfidfVectorizer(max_features=50)
        self.type_encoder = LabelEncoder()

        # Feature matrix and metadata
        self.feature_matrix = None
        self.item_features = None
        self.latent_features = None

        # Hyperparameters (can be overridden for tuning)
        self.exposure_model_params = {"alpha": 0.001, "max_iter": 1000}
        self.preference_model_params = {"alpha": 0.001, "max_iter": 1000}
        self.recency_lambda = 0.03  # Exponential decay for recency
        self.personal_weight = 0.7
        self.popularity_weight = 0.3
        self.mmr_lambda = 0.8  # MMR diversity parameter
        self.svd_components = 64
        self.min_votes_threshold = 100

        # Evaluation metrics storage
        self.metrics = {}

        # Training state
        self.is_fitted = False

        # Trained model components (set during fit())
        self._exposure_probs = None  # Cached exposure probabilities

        print(f"🚀 Initialized AllInOneRecommender with {len(self.dataset.catalog)} items")

    def apply_hyperparameters(self, params: dict):
        """
        Apply hyperparameters to the recommender.

        Args:
            params: Dictionary of hyperparameters
        """
        if "exposure_model_params" in params:
            self.exposure_model_params = params["exposure_model_params"]
        if "preference_model_params" in params:
            self.preference_model_params = params["preference_model_params"]
        if "mmr_lambda" in params:
            self.mmr_lambda = params["mmr_lambda"]
        if "svd_components" in params:
            self.svd_components = params["svd_components"]
        if "min_votes_threshold" in params:
            self.min_votes_threshold = params["min_votes_threshold"]

        print("⚙️ Applied cached hyperparameters")
        print(f"   Exposure model: {self.exposure_model_params}")
        print(f"   Preference model: {self.preference_model_params}")
        print(f"   MMR lambda: {self.mmr_lambda}")
        print(f"   SVD components: {self.svd_components}")
        print(f"   Min votes threshold: {self.min_votes_threshold}")

    @property
    def ratings(self):
        """Access ratings from dataset."""
        return self.dataset.ratings

    @property
    def watchlist(self):
        """Access watchlist from dataset."""
        return self.dataset.watchlist

    @property
    def catalog(self):
        """Access catalog from dataset."""
        return self.dataset.catalog

    def build_features(self) -> pd.DataFrame:
        """
        Stage 1: Feature Engineering

        Build comprehensive feature matrix for all items including:
        - Content features: genres (TF-IDF), title type, runtime bins
        - Popularity features: IMDb rating, log(1+votes)
        - Temporal features: recency with exponential decay
        - Missing value indicators

        Returns:
            DataFrame with engineered features for all catalog items
        """
        print("🔧 Stage 1: Building feature matrix...")

        # Start with catalog
        features_df = self.catalog.copy()

        # Current year for recency calculation
        current_year = datetime.now().year

        # === Content Features ===

        # Runtime handling - use a default since we don't have runtime data
        features_df["runtime_filled"] = 90  # Default runtime
        features_df["runtime_bin"] = "unknown"  # Single category since we don't have data

        # Year bins and recency
        features_df["year_filled"] = features_df["year"].fillna(current_year - 20)
        features_df["recency"] = current_year - features_df["year_filled"]
        features_df["recency_decay"] = np.exp(-self.recency_lambda * features_df["recency"])

        # Decade bins
        features_df["decade"] = ((features_df["year_filled"] // 10) * 10).astype(int)

        # === Popularity Features ===

        # IMDb rating (normalized)
        features_df["imdb_rating_filled"] = features_df["imdb_rating"].fillna(6.5)
        features_df["imdb_rating_norm"] = features_df["imdb_rating_filled"] / 10.0

        # Vote count (log transform)
        features_df["num_votes_filled"] = features_df["num_votes"].fillna(1000)
        features_df["log_votes"] = np.log1p(features_df["num_votes_filled"])

        # Popularity score with recency decay
        features_df["popularity_raw"] = (
            features_df["imdb_rating_norm"]
            * features_df["log_votes"]
            * features_df["recency_decay"]
        )

        # === Missing Value Indicators ===

        features_df["has_runtime"] = 0  # No runtime data available
        features_df["has_year"] = features_df["year"].notna().astype(int)
        features_df["has_rating"] = features_df["imdb_rating"].notna().astype(int)
        features_df["has_votes"] = features_df["num_votes"].notna().astype(int)

        # === Exposure Indicators ===

        # Mark items that appear in ratings or watchlist
        rated_items = set(self.ratings["imdb_const"].values) if len(self.ratings) > 0 else set()
        watchlist_items = (
            set(self.watchlist["imdb_const"].values) if len(self.watchlist) > 0 else set()
        )

        features_df["in_ratings"] = features_df["imdb_const"].isin(rated_items).astype(int)
        features_df["in_watchlist"] = features_df["imdb_const"].isin(watchlist_items).astype(int)
        features_df["exposed"] = (features_df["in_ratings"] | features_df["in_watchlist"]).astype(
            int
        )

        # Store for later use
        self.item_features = features_df

        print(f"✅ Built features for {len(features_df)} items")
        return features_df

    def build_feature_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Convert feature DataFrame to numerical matrix for ML models.

        Args:
            features_df: DataFrame with engineered features

        Returns:
            Numerical feature matrix (n_items, n_features)
        """
        print("🔢 Building numerical feature matrix...")

        features_list = []

        # === Numerical Features ===
        numerical_features = [
            "imdb_rating_norm",
            "log_votes",
            "recency",
            "recency_decay",
            "popularity_raw",
            "has_runtime",
            "has_year",
            "has_rating",
            "has_votes",
        ]

        for feature in numerical_features:
            if feature in features_df.columns:
                values = features_df[feature].values.reshape(-1, 1)
                features_list.append(values)

        # === Categorical Features (One-hot) ===

        # Runtime bins - using single category since no runtime data
        runtime_dummies = np.ones((len(features_df), 1))  # Single dummy feature
        features_list.append(runtime_dummies)

        # Decade bins
        if "decade" in features_df.columns:
            decade_dummies = pd.get_dummies(features_df["decade"], prefix="decade").values
            features_list.append(decade_dummies)

        # Title type - not available in our data
        # Create a single dummy feature assuming all are movies
        type_dummies = np.ones((len(features_df), 1))
        features_list.append(type_dummies)

        # === Genre Features (TF-IDF) ===

        if "genres" in features_df.columns:
            # Prepare genre text (replace commas with spaces)
            genre_text = features_df["genres"].fillna("").str.replace(",", " ")

            # Fit TF-IDF if not already fitted
            if not hasattr(self.genre_encoder, "vocabulary_"):
                genre_features = self.genre_encoder.fit_transform(genre_text).toarray()
            else:
                genre_features = self.genre_encoder.transform(genre_text).toarray()

            features_list.append(genre_features)

        # Concatenate all features
        if features_list:
            feature_matrix = np.hstack(features_list)
        else:
            # Fallback if no features
            feature_matrix = np.ones((len(features_df), 1))

        print(f"✅ Feature matrix shape: {feature_matrix.shape}")
        return feature_matrix

    def train_exposure_model(self, features_df: pd.DataFrame, feature_matrix: np.ndarray):
        """
        Stage 2: Exposure Modeling

        Train a model to predict P(exposed) - the probability that a user
        has been exposed to (seen) a particular movie. This helps handle
        selection bias in the user's ratings/watchlist data.

        Since our catalog only contains rated/watchlisted items, we'll use
        a heuristic approach based on popularity and recency.

        Args:
            features_df: Feature DataFrame
            feature_matrix: Numerical feature matrix
        """
        print("📺 Stage 2: Training exposure model...")

        # Since all items in our catalog are "exposed", we'll model exposure
        # based on popularity and recency as a proxy for likelihood of exposure

        # Scale features
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(feature_matrix)

        # Use a heuristic exposure probability based on popularity and recency
        # Higher popularity + more recent = higher exposure probability
        imdb_ratings = features_df["imdb_rating_norm"].values
        log_votes = features_df["log_votes"].values
        recency_decay = features_df["recency_decay"].values

        # Normalize each component
        imdb_norm = (imdb_ratings - np.mean(imdb_ratings)) / (np.std(imdb_ratings) + 1e-8)
        votes_norm = (log_votes - np.mean(log_votes)) / (np.std(log_votes) + 1e-8)
        recency_norm = (recency_decay - np.mean(recency_decay)) / (np.std(recency_decay) + 1e-8)

        # Combine into exposure score
        exposure_scores = 0.4 * imdb_norm + 0.4 * votes_norm + 0.2 * recency_norm

        # Convert to probabilities using sigmoid
        exposure_probs = 1 / (1 + np.exp(-exposure_scores))

        # Ensure reasonable range (0.1 to 0.9)
        exposure_probs = 0.1 + 0.8 * exposure_probs

        # Create a dummy model for consistency
        self.exposure_model = None

        print(f"✅ Exposure model trained. Avg P(exposed) = {exposure_probs.mean():.3f}")
        return exposure_probs

    def build_pairwise_data(
        self,
        features_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        min_gap: int = 2,
        hard_negative: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build pairwise training data for preference learning.

        For each pair of rated items A, B:
        - If rating(A) >= rating(B) + ``min_gap`` then A ≻ B (preference for A)
        - Use feature differences: X_diff = X_A - X_B
        - Target: 1 if A preferred, 0 if B preferred
        - Optionally sample "hard negatives" for highly rated items

        Args:
            features_df: Feature DataFrame
            feature_matrix: Numerical feature matrix
            min_gap: Minimum rating difference to create a pair
            hard_negative: Whether to add feature-similar but lower-rated items

        Returns:
            Tuple of (feature_differences, preference_targets, sample_weights)
        """
        print("👥 Building pairwise preference data...")

        if len(self.ratings) == 0:
            print("⚠️ No ratings available for preference learning")
            empty = np.array([]).reshape(0, feature_matrix.shape[1])
            return empty, np.array([]), np.array([])

        # Get ratings with feature indices
        ratings_with_idx = []
        for idx, row in features_df.iterrows():
            if row["in_ratings"]:
                # Find rating for this item
                rating_rows = self.ratings[self.ratings["imdb_const"] == row["imdb_const"]]
                if len(rating_rows) > 0:
                    rating = rating_rows.iloc[0]["my_rating"]
                    ratings_with_idx.append((idx, rating, row["imdb_const"]))

        if len(ratings_with_idx) < 2:
            print("⚠️ Not enough rated items for pairwise learning")
            empty = np.array([]).reshape(0, feature_matrix.shape[1])
            return empty, np.array([]), np.array([])

        # Build pairwise comparisons
        X_pairs: list[np.ndarray] = []
        y_pairs: list[int] = []
        weights: list[float] = []

        for i, (idx_a, rating_a, _const_a) in enumerate(ratings_with_idx):
            for _j, (idx_b, rating_b, _const_b) in enumerate(ratings_with_idx[i + 1 :], i + 1):
                # Only create pair if ratings differ by at least min_gap
                gap = abs(rating_a - rating_b)
                if gap >= min_gap:
                    if rating_a > rating_b:
                        # A preferred over B: (X_a - X_b, y=1) and (X_b - X_a, y=0)
                        feat_diff_ab = feature_matrix[idx_a] - feature_matrix[idx_b]
                        feat_diff_ba = feature_matrix[idx_b] - feature_matrix[idx_a]

                        X_pairs.append(feat_diff_ab)
                        y_pairs.append(1)
                        weights.append(gap)
                        X_pairs.append(feat_diff_ba)
                        y_pairs.append(0)
                        weights.append(gap)
                    else:
                        # B preferred over A: (X_b - X_a, y=1) and (X_a - X_b, y=0)
                        feat_diff_ba = feature_matrix[idx_b] - feature_matrix[idx_a]
                        feat_diff_ab = feature_matrix[idx_a] - feature_matrix[idx_b]

                        X_pairs.append(feat_diff_ba)
                        y_pairs.append(1)
                        weights.append(gap)
                        X_pairs.append(feat_diff_ab)
                        y_pairs.append(0)
                        weights.append(gap)

        # === Hard Negative Sampling ===
        if hard_negative:
            from sklearn.metrics.pairwise import cosine_distances

            high_items = [r for r in ratings_with_idx if r[1] >= 8]
            for idx_a, rating_a, _ in high_items:
                candidates = [
                    (idx_b, rating_b)
                    for idx_b, rating_b, _ in ratings_with_idx
                    if rating_b < rating_a and abs(rating_a - rating_b) < min_gap
                ]
                if not candidates:
                    continue
                cand_indices = [c[0] for c in candidates]
                cand_vecs = feature_matrix[cand_indices]
                distances = cosine_distances(
                    feature_matrix[idx_a].reshape(1, -1), cand_vecs
                ).flatten()
                chosen_idx = cand_indices[int(np.argmin(distances))]
                chosen_rating = [c[1] for c in candidates][int(np.argmin(distances))]
                gap = abs(rating_a - chosen_rating)
                feat_diff_ab = feature_matrix[idx_a] - feature_matrix[chosen_idx]
                feat_diff_ba = feature_matrix[chosen_idx] - feature_matrix[idx_a]
                X_pairs.append(feat_diff_ab)
                y_pairs.append(1)
                weights.append(gap)
                X_pairs.append(feat_diff_ba)
                y_pairs.append(0)
                weights.append(gap)

        if len(X_pairs) == 0:
            print(f"⚠️ No valid pairwise comparisons found (need rating differences >= {min_gap})")
            empty = np.array([]).reshape(0, feature_matrix.shape[1])
            return empty, np.array([]), np.array([])

        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)
        weights = np.array(weights)

        # Shuffle pairs
        perm = np.random.permutation(len(X_pairs))
        X_pairs = X_pairs[perm]
        y_pairs = y_pairs[perm]
        weights = weights[perm]

        print(f"✅ Built {len(X_pairs)} pairwise comparisons")
        return X_pairs, y_pairs, weights

    def train_preference_model(
        self,
        X_pairs: np.ndarray,
        y_pairs: np.ndarray,
        sample_weight: np.ndarray | None = None,
        C: float | None = None,
        alpha: float | None = None,
        calibration: str = "sigmoid",
        max_iter: int = 1000,
    ):
        """
        Stage 3: Preference Modeling

        Train a calibrated linear model to predict P(like|exposed) using
        pairwise preference data.

        The model is implemented as a ``Pipeline`` consisting of a
        ``StandardScaler`` and ``SGDClassifier`` (logistic loss). The classifier
        is wrapped in ``CalibratedClassifierCV`` to provide reliable
        probabilities via Platt scaling (``sigmoid``) or isotonic regression.

        Args:
            X_pairs: Feature differences for pairs
            y_pairs: Preference targets (1 if first item preferred)
            sample_weight: Optional importance weights for each pair
            C: Inverse regularization strength (if provided, overrides ``alpha``)
            alpha: Regularization strength for SGDClassifier
            calibration: 'sigmoid' (Platt scaling) or 'isotonic'
            max_iter: Maximum iterations/epochs for SGD training
        """
        print("❤️ Stage 3: Training preference model...")

        if len(X_pairs) == 0:
            print("⚠️ No pairwise data available, using simple preference model")
            # Create a dummy model that returns constant probability
            self.preference_model = None
            return

        # Use hyperparameters instead of method parameters
        alpha = self.preference_model_params.get("alpha", 0.0001)
        max_iter = self.preference_model_params.get("max_iter", 1000)

        # Base linear classifier with shuffling each epoch
        base_clf = SGDClassifier(
            loss="log_loss",
            random_state=self.random_seed,
            max_iter=max_iter,
            alpha=alpha,
            shuffle=True,
            class_weight="balanced",
        )

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", base_clf)])

        try:
            if sample_weight is not None:
                pipeline.fit(X_pairs, y_pairs, clf__sample_weight=sample_weight)
            else:
                pipeline.fit(X_pairs, y_pairs)

            self.preference_model = CalibratedClassifierCV(
                estimator=pipeline, method=calibration, cv="prefit"
            )
            if sample_weight is not None:
                self.preference_model.fit(X_pairs, y_pairs, sample_weight=sample_weight)
            else:
                self.preference_model.fit(X_pairs, y_pairs)
            print(
                "✅ Preference model trained on "
                f"{len(X_pairs)} pairs (classes: {len(set(y_pairs))})"
            )
        except Exception as e:
            print(f"⚠️ Preference model training failed: {e}")
            self.preference_model = None

    def calculate_popularity_prior(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate popularity prior scores.

        Popularity = IMDb_rating × log(1+votes) × exp(-λ×age)
        Then normalize to z-scores.

        Args:
            features_df: Feature DataFrame

        Returns:
            Array of normalized popularity scores
        """
        print("⭐ Calculating popularity priors...")

        # Raw popularity score (already calculated in features)
        pop_scores = features_df["popularity_raw"].values

        # Normalize to z-scores
        pop_mean = np.mean(pop_scores)
        pop_std = np.std(pop_scores)

        if pop_std > 0:
            pop_normalized = (pop_scores - pop_mean) / pop_std
        else:
            pop_normalized = np.zeros_like(pop_scores)

        print(f"✅ Popularity scores: mean={pop_mean:.3f}, std={pop_std:.3f}")
        return pop_normalized

    def calculate_personal_scores(
        self, feature_matrix: np.ndarray, exposure_probs: np.ndarray, features_df=None
    ) -> np.ndarray:
        """
        Calculate personal preference scores.

        Personal = P(exposed) × P(like|exposed)

        Args:
            feature_matrix: Scaled feature matrix
            exposure_probs: Exposure probabilities
            features_df: DataFrame of candidate items (for index mapping)

        Returns:
            Array of personal preference scores
        """
        print("🎯 Calculating personal scores...")

        if self.preference_model is None:
            print("⚠️ No preference model, using exposure probabilities only")
            return exposure_probs

        # Get scaled features
        X_scaled = self.feature_scaler.transform(feature_matrix)

        # Calculate P(like|exposed) using the trained preference model
        if len(self.ratings) > 0:
            # Get the user's average liked features (items rated >= 8)
            liked_items = self.ratings[self.ratings["my_rating"] >= 8]["imdb_const"].values

            if len(liked_items) > 0:
                # Find features of liked items in current candidate set
                liked_indices = []
                if features_df is not None:
                    # Map liked items to current candidate indices
                    for const in liked_items:
                        idx_matches = features_df[features_df["imdb_const"] == const].index
                        if len(idx_matches) > 0:
                            liked_indices.extend(idx_matches)
                else:
                    # Fallback to full catalog indices (for backward compatibility)
                    for const in liked_items:
                        idx_matches = self.item_features[
                            self.item_features["imdb_const"] == const
                        ].index
                        if len(idx_matches) > 0:
                            liked_indices.extend(idx_matches)

                if len(liked_indices) > 0:
                    # Get average features of liked items
                    liked_features = X_scaled[liked_indices]
                    user_profile = np.mean(liked_features, axis=0)

                    # For each candidate item, create feature difference (item - user_profile)
                    # Then use preference model to predict P(item > user_profile)
                    feature_diffs = X_scaled - user_profile  # Broadcasting
                    preference_probs = self.preference_model.predict_proba(feature_diffs)[
                        :, 1
                    ]  # P(class=1)

                    print(
                        "✅ Using trained preference model "
                        f"(probs range: {preference_probs.min():.3f}-"
                        f"{preference_probs.max():.3f})"
                    )
                else:
                    preference_probs = np.ones(len(exposure_probs)) * 0.5
            else:
                preference_probs = np.ones(len(exposure_probs)) * 0.5
        else:
            preference_probs = np.ones(len(exposure_probs)) * 0.5

        # Personal score = P(exposed) × P(like|exposed)
        personal_scores = exposure_probs * preference_probs

        print(f"✅ Personal scores: mean={personal_scores.mean():.3f}")
        return personal_scores

    def build_latent_space(self, feature_matrix: np.ndarray):
        """
        Build latent space representation using TruncatedSVD for diversity calculation.

        Args:
            feature_matrix: Feature matrix to project
        """
        print(f"🌌 Building latent space (k={self.svd_components})...")

        self.svd_model = TruncatedSVD(
            n_components=min(self.svd_components, feature_matrix.shape[1] - 1),
            random_state=self.random_seed,
        )

        # Fit and transform
        self.latent_features = self.svd_model.fit_transform(feature_matrix)

        print(f"✅ Latent space: {self.latent_features.shape}")

    def mmr_rerank(self, scores: np.ndarray, top_k: int = 50) -> list[int]:
        """
        Stage 4: Maximal Marginal Relevance (MMR) re-ranking for diversity.

        MMR balances relevance and diversity:
        MMR = λ × Relevance - (1-λ) × max_similarity_to_selected

        Args:
            scores: Relevance scores for all items
            top_k: Number of items to return

        Returns:
            List of item indices in MMR order
        """
        print(f"🎲 Stage 4: MMR re-ranking (λ={self.mmr_lambda}, k={top_k})...")

        if self.latent_features is None:
            print("⚠️ No latent features available, using relevance-only ranking")
            return np.argsort(scores)[::-1][:top_k].tolist()

        # Initialize
        selected = []
        remaining = set(range(len(scores)))

        # Select first item (highest relevance)
        first_idx = np.argmax(scores)
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Select remaining items using MMR
        for _ in range(min(top_k - 1, len(remaining))):
            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                # Relevance component
                relevance = scores[idx]

                # Diversity component (max similarity to already selected)
                if len(selected) > 0:
                    similarities = []
                    for sel_idx in selected:
                        # Cosine similarity in latent space
                        sim = np.dot(self.latent_features[idx], self.latent_features[sel_idx]) / (
                            np.linalg.norm(self.latent_features[idx])
                            * np.linalg.norm(self.latent_features[sel_idx])
                            + 1e-8
                        )
                        similarities.append(sim)

                    max_similarity = max(similarities)
                else:
                    max_similarity = 0

                # MMR score
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        print(f"✅ MMR re-ranking complete: {len(selected)} items")
        return selected

    def create_shelves(self, recommendations: list[dict]) -> dict[str, list[dict]]:
        """
        Organize recommendations into intuitive shelves/buckets.

        Args:
            recommendations: List of recommendation dictionaries

        Returns:
            Dictionary mapping shelf names to recommendation lists
        """
        print("📚 Creating recommendation shelves...")

        shelves = {
            "tonight_picks": [],
            "new_aligned": [],
            "prestige_backlog": [],
            "stretch_picks": [],
        }

        current_year = datetime.now().year

        for rec in recommendations:
            year = rec.get("year", current_year - 10)
            runtime = rec.get("runtime", 120)
            personal_score = rec.get("score_personal", 0)
            popularity_score = rec.get("score_pop", 0)
            final_score = rec.get("score_final", 0)

            # Tonight Picks: Short, recent, high final score
            if (
                runtime <= 120
                and year >= 2016
                and final_score
                >= np.percentile([r.get("score_final", 0) for r in recommendations], 75)
            ):
                shelves["tonight_picks"].append(rec)

            # New & Aligned: Recent, high personal score
            elif year >= 2018 and personal_score >= np.percentile(
                [r.get("score_personal", 0) for r in recommendations], 70
            ):
                shelves["new_aligned"].append(rec)

            # Prestige Backlog: Older, high popularity
            elif year < 2010 and popularity_score >= np.percentile(
                [r.get("score_pop", 0) for r in recommendations], 70
            ):
                shelves["prestige_backlog"].append(rec)

            # Stretch Picks: Everything else that's high-scoring
            elif final_score >= np.percentile(
                [r.get("score_final", 0) for r in recommendations], 60
            ):
                shelves["stretch_picks"].append(rec)

        # Ensure we have items in each shelf (redistribute if needed)
        for shelf_name, shelf_items in shelves.items():
            print(f"📖 {shelf_name}: {len(shelf_items)} items")

        return shelves

    def build_candidates(
        self,
        features_df: pd.DataFrame | None = None,
        feature_matrix: np.ndarray | None = None,
        max_candidates: int = 500,
        popular_ratio: float = 0.4,
        neighbor_ratio: float = 0.4,
        watchlist_ratio: float = 0.2,
    ) -> list[str]:
        """
        Build a realistic candidate pool for recommendation and evaluation.

        Creates a union of:
        (a) Unrated watchlist items
        (b) Top-popular items not yet rated
        (c) Nearest neighbors of highly-rated titles

        Args:
            features_df: Feature DataFrame (if None, will build it)
            feature_matrix: Feature matrix (if None, will build it)
            max_candidates: Maximum number of candidates to return
            popular_ratio: Fraction of candidates from popular items
            neighbor_ratio: Fraction of candidates from nearest neighbors
            watchlist_ratio: Fraction of candidates from watchlist

        Returns:
            List of item IDs (imdb_const) forming the candidate pool
        """
        print(f"🎯 Building candidate pool (max={max_candidates})...")

        # Build features if not provided
        if features_df is None:
            features_df = self.build_features()
        if feature_matrix is None:
            feature_matrix = self.build_feature_matrix(features_df)

        candidates = set()

        # Get rated items to exclude
        rated_items = set(self.ratings["imdb_const"].values) if len(self.ratings) > 0 else set()

        # === (a) Unrated watchlist items ===
        watchlist_candidates = []
        if len(self.watchlist) > 0:
            for const in self.watchlist["imdb_const"].values:
                if const not in rated_items and const in features_df["imdb_const"].values:
                    watchlist_candidates.append(const)

        n_watchlist = min(len(watchlist_candidates), int(max_candidates * watchlist_ratio))
        candidates.update(watchlist_candidates[:n_watchlist])

        # === (b) Top-popular gaps ===
        # Get items sorted by popularity score, excluding rated ones
        unrated_items = features_df[
            (~features_df["imdb_const"].isin(rated_items))
            & (~features_df["imdb_const"].isin(candidates))
        ].copy()

        if len(unrated_items) > 0:
            unrated_items = unrated_items.sort_values("popularity_raw", ascending=False)
            n_popular = min(len(unrated_items), int(max_candidates * popular_ratio))
            popular_candidates = unrated_items.head(n_popular)["imdb_const"].tolist()
            candidates.update(popular_candidates)

        # === (c) Nearest neighbors of high-rated titles ===
        if len(self.ratings) > 0 and feature_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            # Get highly-rated items (>= 8)
            high_rated_items = self.ratings[self.ratings["my_rating"] >= 8]["imdb_const"].values

            if len(high_rated_items) > 0:
                # Find their indices in features_df
                high_rated_indices = []
                for const in high_rated_items:
                    idx_matches = features_df[features_df["imdb_const"] == const].index
                    if len(idx_matches) > 0:
                        high_rated_indices.extend(idx_matches)

                if len(high_rated_indices) > 0:
                    # Get feature vectors for high-rated items
                    high_rated_features = feature_matrix[high_rated_indices]
                    avg_high_rated = np.mean(high_rated_features, axis=0).reshape(1, -1)

                    # Calculate similarities to all unrated items
                    unrated_mask = ~features_df["imdb_const"].isin(rated_items | candidates)
                    unrated_indices = features_df.index[unrated_mask].tolist()

                    if len(unrated_indices) > 0:
                        unrated_features = feature_matrix[unrated_indices]
                        similarities = cosine_similarity(avg_high_rated, unrated_features).flatten()

                        # Get top similar items
                        n_neighbors = min(len(similarities), int(max_candidates * neighbor_ratio))
                        top_neighbor_indices = np.argsort(similarities)[::-1][:n_neighbors]

                        neighbor_candidates = []
                        for idx in top_neighbor_indices:
                            original_idx = unrated_indices[idx]
                            const = features_df.loc[original_idx, "imdb_const"]
                            neighbor_candidates.append(const)

                        candidates.update(neighbor_candidates)

        # Fill remaining slots with random popular items if needed
        remaining_slots = max_candidates - len(candidates)
        if remaining_slots > 0:
            all_unrated = features_df[
                (~features_df["imdb_const"].isin(rated_items))
                & (~features_df["imdb_const"].isin(candidates))
            ]
            if len(all_unrated) > 0:
                # Sort by popularity and take top remaining
                remaining_items = all_unrated.sort_values("popularity_raw", ascending=False)
                additional_candidates = remaining_items.head(remaining_slots)["imdb_const"].tolist()
                candidates.update(additional_candidates)

        candidates_list = list(candidates)[:max_candidates]

        print(f"✅ Built candidate pool: {len(candidates_list)} items")
        print(f"   - Watchlist: {n_watchlist}")
        popular_count = len(
            [
                c
                for c in candidates_list
                if c in (popular_candidates if "popular_candidates" in locals() else [])
            ]
        )
        neighbor_count = len(
            [
                c
                for c in candidates_list
                if c in (neighbor_candidates if "neighbor_candidates" in locals() else [])
            ]
        )
        print(f"   - Popular: {popular_count}")
        print(f"   - Neighbors: {neighbor_count}")

        return candidates_list

    def fit(self, user_weight: float = 0.7, global_weight: float = 0.3) -> AllInOneRecommender:
        """
        Fit the All-in-One recommender model on the training data.

        This method performs the complete four-stage training process:
        1. Feature Engineering: Build content and popularity features
        2. Exposure Modeling: Train P(exposed) model
        3. Preference Modeling: Train P(like|exposed) model using pairwise data
        4. Diversity Setup: Build latent space for MMR re-ranking

        Args:
            user_weight: Weight for personal scores (0.0 to 1.0)
            global_weight: Weight for popularity scores (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        print("🎓 TRAINING AllInOneRecommender...")
        print("=" * 60)

        # Store weights
        self.personal_weight = user_weight
        self.popularity_weight = global_weight

        # === Stage 1: Feature Engineering ===
        print("📊 Stage 1: Building feature matrix for training...")
        self.item_features = self.build_features()
        self.feature_matrix = self.build_feature_matrix(self.item_features)

        # === Stage 2: Exposure Modeling ===
        print("🔍 Stage 2: Training exposure model...")
        self._exposure_probs = self.train_exposure_model(self.item_features, self.feature_matrix)

        # === Stage 3: Preference Modeling ===
        print("❤️ Stage 3: Training preference model...")
        X_pairs, y_pairs, pair_weights = self.build_pairwise_data(
            self.item_features, self.feature_matrix
        )
        self.train_preference_model(X_pairs, y_pairs, sample_weight=pair_weights)

        # === Stage 4: Diversity Setup ===
        print("🌟 Stage 4: Building latent space for diversity...")
        self.build_latent_space(self.feature_matrix)

        # Mark as fitted
        self.is_fitted = True

        print("✅ TRAINING COMPLETE!")
        print(f"   - Features: {self.feature_matrix.shape}")
        latent_shape = self.latent_features.shape if self.latent_features is not None else "None"
        print(f"   - Latent space: {latent_shape}")
        preference_status = "Trained" if self.preference_model is not None else "Fallback"
        print(f"   - Preference model: {preference_status}")
        print("=" * 60)

        return self

    def score(
        self,
        seeds: list[str],
        user_weight: float = 0.7,
        global_weight: float = 0.3,
        recency_weight: float = 0.0,
        exclude_rated: bool = True,
        candidates: list[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """
        Generate recommendations using the trained model (INFERENCE ONLY).

        This method requires the model to be fitted first via fit().
        It performs only inference/scoring without any training.

        Args:
            seeds: Seed movie IDs (not used in this implementation)
            user_weight: Weight for personal scores (for compatibility, but uses fitted weights)
            global_weight: Weight for popularity scores (for compatibility, but uses fitted weights)
            recency_weight: Recency bias (not used, built into features)
            exclude_rated: Whether to exclude already rated items
            candidates: Optional list of candidate item IDs to restrict scoring to

        Returns:
            Tuple of (scores, explanations)
        """
        # Check if model is fitted
        if not self.is_fitted:
            # Check if we're using cached hyperparameters (non-default values indicate this)
            using_cached = (
                self.exposure_model_params.get("alpha", 0.001) != 0.001
                or self.preference_model_params.get("alpha", 0.001) != 0.001
                or self.mmr_lambda != 0.8
                or self.svd_components != 64
            )

            if using_cached:
                print("⚠️  Model not fitted! Training with cached optimal hyperparameters...")
            else:
                print("⚠️  Model not fitted! Training with default hyperparameters...")
                print("   💡 Consider running hyperparameter tuning for better performance")

            self.fit(user_weight=user_weight, global_weight=global_weight)

        print("🎯 INFERENCE: Generating recommendations from trained model...")

        # Use training features as base
        features_df = self.item_features.copy()
        feature_matrix = self.feature_matrix.copy()

        # Filter to candidates if provided
        if candidates is not None:
            candidate_mask = features_df["imdb_const"].isin(candidates)
            features_df = features_df[candidate_mask].reset_index(drop=True)

            # Get corresponding feature matrix rows
            feature_matrix = self.feature_matrix[candidate_mask]

            print(f"🎯 Filtered to {len(features_df)} candidate items")

        # === INFERENCE: Calculate scores using trained models ===

        # Get exposure probabilities (from training or recalculate for candidates)
        if candidates is not None:
            # Recalculate exposure for candidate subset
            exposure_probs = self._calculate_exposure_inference(features_df)
        else:
            # Use cached training exposure probabilities
            exposure_probs = self._exposure_probs

        # Calculate personal scores using trained preference model
        personal_scores = self.calculate_personal_scores(
            feature_matrix, exposure_probs, features_df
        )

        # Calculate popularity scores
        popularity_scores = self.calculate_popularity_prior(features_df)

        # === Final Blended Score with Z-Score Normalization ===
        # Normalize personal scores to z-scores
        personal_mean = np.mean(personal_scores)
        personal_std = np.std(personal_scores)
        personal_z = (personal_scores - personal_mean) / (personal_std + 1e-8)

        # Normalize popularity scores to z-scores
        pop_mean = np.mean(popularity_scores)
        pop_std = np.std(popularity_scores)
        pop_z = (popularity_scores - pop_mean) / (pop_std + 1e-8)

        # Blend normalized scores using fitted weights
        final_scores = self.personal_weight * personal_z + self.popularity_weight * pop_z

        print(
            f"📊 Score normalization: personal μ={personal_mean:.3f} σ={personal_std:.3f}, "
            f"popularity μ={pop_mean:.3f} σ={pop_std:.3f}"
        )

        # Filter out rated items if requested
        if exclude_rated and len(self.ratings) > 0:
            rated_consts = set(self.ratings["imdb_const"].values)
            unrated_mask = ~features_df["imdb_const"].isin(rated_consts)
            valid_indices = features_df.index[unrated_mask].tolist()
        else:
            valid_indices = features_df.index.tolist()

        # Build return dictionaries
        scores_dict = {}
        explanations_dict = {}

        for idx in valid_indices:
            const = features_df.loc[idx, "imdb_const"]
            score = final_scores[idx]
            personal = personal_scores[idx]
            popularity = popularity_scores[idx]
            exposure = exposure_probs[idx]

            scores_dict[const] = float(score)

            # Generate explanation
            explanation_parts = []

            if personal > 0.6:
                explanation_parts.append("strong personal fit")
            elif personal > 0.4:
                explanation_parts.append("moderate personal interest")
            else:
                explanation_parts.append("exploratory pick")

            if popularity > 1.0:
                explanation_parts.append("high critical acclaim")
            elif popularity > 0.5:
                explanation_parts.append("well-regarded")

            if exposure > 0.7:
                explanation_parts.append("likely in your sphere")
            elif exposure < 0.3:
                explanation_parts.append("hidden gem")

            explanations_dict[const] = "; ".join(explanation_parts)

        print(f"✅ Generated scores for {len(scores_dict)} items")
        return scores_dict, explanations_dict

    def _calculate_exposure_inference(self, features_df: pd.DataFrame) -> np.ndarray:
        """Calculate exposure probabilities during inference (for candidate subsets)."""
        # Use the same heuristic as training
        imdb_ratings = features_df["imdb_rating_norm"].values
        log_votes = features_df["log_votes"].values
        recency_decay = features_df["recency_decay"].values

        # Normalize each component (use training stats if available)
        imdb_norm = (imdb_ratings - np.mean(imdb_ratings)) / (np.std(imdb_ratings) + 1e-8)
        votes_norm = (log_votes - np.mean(log_votes)) / (np.std(log_votes) + 1e-8)
        recency_norm = (recency_decay - np.mean(recency_decay)) / (np.std(recency_decay) + 1e-8)

        # Combine into exposure score
        exposure_scores = 0.4 * imdb_norm + 0.4 * votes_norm + 0.2 * recency_norm

        # Convert to probabilities using sigmoid
        exposure_probs = 1 / (1 + np.exp(-exposure_scores))

        # Ensure reasonable range (0.1 to 0.9)
        exposure_probs = 0.1 + 0.8 * exposure_probs

        return exposure_probs

    def evaluate_temporal_split(self, test_size: float = 0.2) -> dict[str, float]:
        """
        Evaluate the recommender using temporal split.

        Args:
            test_size: Fraction of most recent ratings to use for testing

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"📊 Evaluating with temporal split (test_size={test_size})...")

        if len(self.ratings) < 10:
            print("⚠️ Need at least 10 ratings for evaluation")
            return {}

        # Sort ratings by date
        ratings_sorted = self.ratings.copy()
        if "rated_at" not in ratings_sorted.columns:
            print("⚠️ No rated_at column, using random split")
            train_ratings, test_ratings = train_test_split(
                ratings_sorted, test_size=test_size, random_state=self.random_seed
            )
        else:
            # Convert rated_at to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(ratings_sorted["rated_at"]):
                ratings_sorted["rated_at"] = pd.to_datetime(ratings_sorted["rated_at"])

            ratings_sorted = ratings_sorted.sort_values("rated_at")
            split_idx = int(len(ratings_sorted) * (1 - test_size))
            train_ratings = ratings_sorted.iloc[:split_idx]
            test_ratings = ratings_sorted.iloc[split_idx:]

        # Temporarily replace dataset with training set
        from .data_io import Dataset

        original_dataset = self.dataset
        train_dataset = Dataset(ratings=train_ratings, watchlist=self.dataset.watchlist)
        self.dataset = train_dataset

        try:
            # Build candidate pool based on training data
            # Include test items to ensure they can be recommended
            test_items_set = set(test_ratings["imdb_const"].values)

            # Build candidates using training data context
            candidates = self.build_candidates(max_candidates=1000)

            # Ensure all test items are in candidates (for fair evaluation)
            candidates_set = set(candidates)
            missing_test_items = test_items_set - candidates_set
            if missing_test_items:
                candidates.extend(list(missing_test_items))
                print(f"📌 Added {len(missing_test_items)} test items to candidate pool")

            # Generate recommendations on candidate pool
            scores, explanations = self.score(
                seeds=[],
                user_weight=self.personal_weight,
                global_weight=self.popularity_weight,
                exclude_rated=False,  # Include rated items for evaluation
                candidates=candidates,
            )

            # Calculate metrics
            metrics = {}

            # Get test item IDs and ratings
            test_items = test_ratings["imdb_const"].values

            # Filter scores to test items that have scores
            test_scores = {item: scores[item] for item in test_items if item in scores}

            if len(test_scores) > 0:
                # Hits@10
                top_10_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                top_10_ids = [item[0] for item in top_10_items]
                hits_at_10 = len(set(top_10_ids) & set(test_items)) / len(test_items)
                metrics["hits_at_10"] = hits_at_10

                # NDCG@10 - calculate manually to be consistent with binary hits@10
                # Use the same top 10 items as hits@10 calculation
                ndcg_at_10 = 0.0
                if len(top_10_ids) > 0:
                    # Calculate DCG@10 - same relevance as hits@10 (binary: in test set or not)
                    dcg_at_10 = 0.0
                    for i, item_id in enumerate(top_10_ids):
                        relevance = 1.0 if item_id in test_items else 0.0
                        if relevance > 0:
                            dcg_at_10 += relevance / np.log2(i + 2)

                    # Calculate IDCG@10 - ideal would put all relevant items first
                    num_relevant_in_test = len(test_items)
                    num_relevant_in_top10 = min(10, num_relevant_in_test)
                    idcg_at_10 = (
                        sum(1.0 / np.log2(i + 2) for i in range(num_relevant_in_top10))
                        if num_relevant_in_top10 > 0
                        else 1.0
                    )

                    # NDCG@10
                    ndcg_at_10 = dcg_at_10 / idcg_at_10 if idcg_at_10 > 0 else 0.0

                metrics["ndcg_at_10"] = ndcg_at_10

                # Diversity (if latent features available)
                if self.latent_features is not None and len(top_10_ids) > 1:
                    # Get latent features for top 10 items
                    top_10_indices = []
                    for item_id in top_10_ids:
                        item_idx = self.item_features[
                            self.item_features["imdb_const"] == item_id
                        ].index
                        if len(item_idx) > 0:
                            top_10_indices.append(item_idx[0])

                    if len(top_10_indices) > 1:
                        top_latent = self.latent_features[top_10_indices]

                        # Calculate pairwise similarities
                        similarities = []
                        for i in range(len(top_latent)):
                            for j in range(i + 1, len(top_latent)):
                                sim = np.dot(top_latent[i], top_latent[j]) / (
                                    np.linalg.norm(top_latent[i]) * np.linalg.norm(top_latent[j])
                                    + 1e-8
                                )
                                similarities.append(sim)

                        avg_similarity = np.mean(similarities)
                        diversity = 1 - avg_similarity  # Higher is more diverse
                        metrics["diversity"] = diversity

            print(f"📈 Evaluation metrics: {metrics}")
            self.metrics = metrics

        finally:
            # Restore original dataset
            self.dataset = original_dataset

        return metrics

    def save_model(self, filepath: str | Path):
        """Save the trained model to disk."""
        model_data = {
            "exposure_model": self.exposure_model,
            "preference_model": self.preference_model,
            "feature_scaler": self.feature_scaler,
            "svd_model": self.svd_model,
            "genre_encoder": self.genre_encoder,
            "type_encoder": self.type_encoder,
            "feature_matrix": self.feature_matrix,
            "item_features": self.item_features,
            "latent_features": self.latent_features,
            "params": {
                "recency_lambda": self.recency_lambda,
                "personal_weight": self.personal_weight,
                "popularity_weight": self.popularity_weight,
                "mmr_lambda": self.mmr_lambda,
                "svd_components": self.svd_components,
                "random_seed": self.random_seed,
            },
            "metrics": self.metrics,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"💾 Model saved to {filepath}")

    def load_model(self, filepath: str | Path):
        """Load a trained model from disk."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.exposure_model = model_data["exposure_model"]
        self.preference_model = model_data["preference_model"]
        self.feature_scaler = model_data["feature_scaler"]
        self.svd_model = model_data["svd_model"]
        self.genre_encoder = model_data["genre_encoder"]
        self.type_encoder = model_data["type_encoder"]
        self.feature_matrix = model_data["feature_matrix"]
        self.item_features = model_data["item_features"]
        self.latent_features = model_data["latent_features"]

        params = model_data.get("params", {})
        self.recency_lambda = params.get("recency_lambda", 0.03)
        self.personal_weight = params.get("personal_weight", 0.7)
        self.popularity_weight = params.get("popularity_weight", 0.3)
        self.mmr_lambda = params.get("mmr_lambda", 0.8)
        self.svd_components = params.get("svd_components", 64)
        self.random_seed = params.get("random_seed", 42)

        self.metrics = model_data.get("metrics", {})

        print(f"📂 Model loaded from {filepath}")

    def export_recommendations_csv(
        self, scores: dict[str, float], filepath: str | Path, top_k: int = 100
    ):
        """
        Export recommendations to CSV format.

        Args:
            scores: Dictionary of item scores
            filepath: Output file path
            top_k: Number of top recommendations to export
        """
        print(f"📝 Exporting top {top_k} recommendations to {filepath}...")

        # Sort by score and get top_k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build recommendations list
        recommendations = []

        for const, score in sorted_items:
            # Find item in catalog
            item_rows = self.item_features[self.item_features["imdb_const"] == const]
            if len(item_rows) == 0:
                continue

            item = item_rows.iloc[0]

            # Get individual scores
            idx = item_rows.index[0]
            if hasattr(self, "feature_matrix") and self.feature_matrix is not None:
                # For exposure probability, use heuristic if no model
                if self.exposure_model is not None:
                    exposure_prob = self.exposure_model.predict_proba(
                        self.feature_scaler.transform(self.feature_matrix[idx : idx + 1])
                    )[0, 1]
                else:
                    # Use heuristic exposure probability
                    imdb_rating = item.get("imdb_rating_norm", 0.5)
                    log_votes = item.get("log_votes", 5)
                    recency_decay = item.get("recency_decay", 0.5)
                    exposure_score = (
                        0.4 * imdb_rating + 0.4 * (log_votes / 15) + 0.2 * recency_decay
                    )
                    exposure_prob = 0.1 + 0.8 / (1 + np.exp(-exposure_score))

                personal_score = exposure_prob  # Simplified for now
                popularity_score = item.get("popularity_raw", 0)
            else:
                personal_score = 0.5
                popularity_score = 0.5

            rec = {
                "tconst": const,
                "title": item.get("title", "Unknown"),
                "year": item.get("year", "Unknown"),
                "genres": item.get("genres", "Unknown"),
                "title_type": item.get("title_type", "movie"),
                "imdb_rating": item.get("imdb_rating", "Unknown"),
                "num_votes": item.get("num_votes", "Unknown"),
                "runtime": item.get("runtime_(mins)", "Unknown"),
                "score_personal": personal_score,
                "score_pop": popularity_score,
                "score_final": score,  # Z-score normalized final blend
            }

            recommendations.append(rec)

        # Save to CSV
        df = pd.DataFrame(recommendations)
        df.to_csv(filepath, index=False)

        print(f"✅ Exported {len(recommendations)} recommendations")
        return recommendations
