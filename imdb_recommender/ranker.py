from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from .schemas import Recommendation
from .data_io import Dataset


class Ranker:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def blend(self, algo_scores: dict[str, dict[str, float]]) -> dict[str, float]:
        out = {}
        for _, scores in algo_scores.items():
            for cid, s in scores.items():
                out.setdefault(cid, []).append(float(s))
        return {cid: float(np.mean(v)) for cid, v in out.items()}

    def top_n(
        self,
        blended: dict[str, float],
        dataset: Dataset,
        topk: int = 25,
        explanations: Optional[dict[str, dict[str, str]]] = None,
        exclude_rated: bool = True,
    ) -> list[Recommendation]:
        rated = set(dataset.ratings["imdb_const"].tolist()) if exclude_rated else set()
        items = [(cid, s) for cid, s in blended.items() if cid not in rated]
        cat = dataset.catalog.set_index("imdb_const", drop=False)

        def safe_float(val):
            if val is None or pd.isna(val):
                return 0.0
            return float(val)

        items.sort(
            key=lambda x: (
                x[1],
                safe_float(cat.loc[x[0]].get("num_votes")),
                safe_float(cat.loc[x[0]].get("year")),
            ),
            reverse=True,
        )
        out = []
        for cid, s in items[:topk]:
            row = cat.loc[cid]
            why_parts = []
            if explanations:
                for expl_map in explanations.values():
                    if cid in expl_map and expl_map[cid]:
                        why_parts.append(expl_map[cid])
            why = "; ".join(why_parts) or "blended score from multiple models"
            out.append(
                Recommendation(
                    imdb_const=cid,
                    title=row.get("title"),
                    year=int(row.get("year")) if pd.notna(row.get("year")) else None,
                    genres=row.get("genres"),
                    score=float(s),
                    why_explainer=why,
                )
            )
        return out
