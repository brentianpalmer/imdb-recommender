from __future__ import annotations

import numpy as np

from .features import content_vector, cosine, recency_weight
from .recommender_base import RecommenderAlgo


class PopSimRecommender(RecommenderAlgo):
    def score(self, seeds, user_weight, global_weight, recency, exclude_rated):
        ds = self.dataset
        cat = ds.catalog.copy()
        cat["vec"] = cat.apply(content_vector, axis=1)
        vecs = dict(zip(cat["imdb_const"], cat["vec"], strict=False))
        liked = ds.ratings[ds.ratings["my_rating"] >= 8]["imdb_const"].tolist()
        base_set = seeds if seeds else liked or ds.ratings["imdb_const"].tolist()
        if not base_set:
            return {}, {}
        centroid = np.mean(
            [vecs.get(t, np.zeros_like(next(iter(vecs.values())))) for t in base_set], axis=0
        )
        pop = cat[["imdb_const", "imdb_rating", "num_votes", "year", "title", "genres"]].copy()
        pop["imdb_rating"] = pop["imdb_rating"].fillna(pop["imdb_rating"].mean())
        pop["num_votes"] = pop["num_votes"].fillna(0)
        r_min, r_max = pop["imdb_rating"].min(), pop["imdb_rating"].max()
        pop["pop_norm"] = (pop["imdb_rating"] - r_min) / (r_max - r_min + 1e-9)
        v_max = pop["num_votes"].max() or 1
        pop["vote_bonus"] = (pop["num_votes"] / v_max) ** 0.5 * 0.1
        pop["global_score"] = (pop["pop_norm"] + pop["vote_bonus"]).clip(0, 1)
        sims = {row["imdb_const"]: cosine(centroid, row["vec"]) for _, row in cat.iterrows()}
        scores, explain = {}, {}
        rated_set = set(ds.ratings["imdb_const"].tolist()) if exclude_rated else set()
        for _, row in pop.iterrows():
            cid = row["imdb_const"]
            if cid in rated_set:
                continue
            s_user = sims.get(cid, 0.0)
            s_global = float(row["global_score"])
            r_weight = recency_weight(row.get("year"), recency)
            score = (user_weight * s_user + global_weight * s_global) * r_weight
            scores[cid] = score
            why = []
            if s_user > 0.6:
                why.append("high genre overlap with your 8–10s")
            elif s_user > 0.3:
                why.append("moderate content similarity to your tastes")
            if s_global > 0.6:
                why.append("strong global acclaim")
            if recency > 0.1 and r_weight > 1.0:
                why.append("recent release bias applied")
            explain[cid] = "; ".join(why) or "balanced blend of your taste and popularity"
        return scores, explain
