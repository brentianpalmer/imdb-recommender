from __future__ import annotations

import numpy as np

from .features import content_vector
from .recommender_base import RecommenderAlgo


class SVDAutoRecommender(RecommenderAlgo):
    def _build_matrix(self, ds):
        cat = ds.catalog.copy()
        items = cat["imdb_const"].tolist()
        item_index = {cid: i for i, cid in enumerate(items)}
        n_items = len(items)
        n_users = 2
        R = np.zeros((n_users, n_items), dtype=float)
        for _, r in ds.ratings.iterrows():
            i = item_index.get(r["imdb_const"])
            if i is not None:
                R[0, i] = float(r["my_rating"])
        if "imdb_rating" in cat.columns:
            R[1, :] = cat["imdb_rating"].fillna(0).to_numpy()
        return (
            R,
            item_index,
            {i: cid for cid, i in item_index.items()},
            cat.set_index("imdb_const", drop=False),
        )

    def _als(self, R, k=8, reg=0.2, iters=25, seed=42):
        np.random.seed(seed)
        m, n = R.shape
        M = (R > 0).astype(float)
        U = 0.1 * np.random.randn(m, k)
        V = 0.1 * np.random.randn(n, k)
        for _ in range(iters):
            for i in range(m):
                Vi = V[M[i, :] > 0]
                Ri = R[i, M[i, :] > 0]
                if Vi.shape[0] == 0:
                    continue
                A = Vi.T @ Vi + reg * np.eye(k)
                b = Vi.T @ Ri
                U[i] = np.linalg.solve(A, b)
            for j in range(n):
                Uj = U[M[:, j] > 0]
                Rj = R[M[:, j] > 0, j]
                if Uj.shape[0] == 0:
                    continue
                A = Uj.T @ Uj + reg * np.eye(k)
                b = Uj.T @ Rj
                V[j] = np.linalg.solve(A, b)
        return U, V

    def score(self, seeds, user_weight, global_weight, recency, exclude_rated):
        R, item_index, index_item, cat = self._build_matrix(self.dataset)
        if R.size == 0:
            return {}, {}
        U, V = self._als(R, k=8, reg=0.2, iters=25, seed=self.random_seed)
        preds = U[0] @ V.T
        rated_set = set(self.dataset.ratings["imdb_const"].tolist()) if exclude_rated else set()
        scores, explain = {}, {}
        if seeds:
            seed_vec = np.mean(
                [content_vector(cat.loc[s]) for s in seeds if s in cat.index], axis=0
            )
        else:
            seed_vec = None
        for j, p in enumerate(preds):
            cid = index_item[j]
            if cid in rated_set:
                continue
            score = max(0.0, min(1.0, (float(p) - 1.0) / 9.0))
            scores[cid] = score
            why = []
            if seed_vec is not None and cid in cat.index:
                cs = float(np.clip(seed_vec @ content_vector(cat.loc[cid]), 0, 1))
                if cs > 0.6:
                    why.append("similar to your seed titles")
            if p >= 8.0:
                why.append("predicted high personal rating")
            explain[cid] = "; ".join(why) or "latent-factor fit from your ratings and IMDb priors"
        return scores, explain
