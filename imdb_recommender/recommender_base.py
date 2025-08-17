
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type
import random
from .data_io import Dataset

class RecommenderAlgo(ABC):
    def __init__(self, dataset: Dataset, random_seed: int = 42):
        self.dataset = dataset; self.random_seed = random_seed; random.seed(random_seed)

    @abstractmethod
    def score(self, seeds: list[str], user_weight: float, global_weight: float, recency: float, exclude_rated: bool):
        ...

class Registry:
    _algos: Dict[str, Type[RecommenderAlgo]] = {}
    @classmethod
    def register(cls, name: str, algo: Type[RecommenderAlgo]):
        cls._algos[name] = algo
    @classmethod
    def get(cls, name: str) -> Type[RecommenderAlgo]:
        if name not in cls._algos: raise KeyError(f"Recommender '{name}' not registered")
        return cls._algos[name]
