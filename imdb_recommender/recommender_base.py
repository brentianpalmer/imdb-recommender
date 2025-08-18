from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .data_io import Dataset


class RecommenderAlgo(ABC):
    def __init__(self, dataset: Dataset, random_seed: int = 42):
        self.dataset = dataset
        self.random_seed = random_seed
        random.seed(random_seed)

    @abstractmethod
    def score(
        self,
        seeds: list[str],
        user_weight: float,
        global_weight: float,
        recency: float,
        exclude_rated: bool,
    ): ...


class Registry:
    _algos: dict[str, type[RecommenderAlgo]] = {}

    @classmethod
    def register(cls, name: str, algo: type[RecommenderAlgo]):
        cls._algos[name] = algo

    @classmethod
    def get(cls, name: str) -> type[RecommenderAlgo]:
        if name not in cls._algos:
            raise KeyError(f"Recommender '{name}' not registered")
        return cls._algos[name]
