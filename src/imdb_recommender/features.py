from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Sport",
    "Thriller",
    "War",
    "Western",
]


def genres_to_vec(genres_str: str | None) -> np.ndarray:
    vec = np.zeros(len(GENRES), dtype=float)
    if not genres_str or not isinstance(genres_str, str):
        return vec
    parts = [g.strip() for g in genres_str.split(",") if g.strip()]
    for p in parts:
        if p in GENRES:
            vec[GENRES.index(p)] = 1.0
    if vec.sum() > 0:
        vec = vec / np.linalg.norm(vec)
    return vec


def year_to_bucket(year):
    edges = [1980, 1990, 2000, 2010, 2020]
    vec = np.zeros(len(edges) + 1, dtype=float)
    if year is None or (isinstance(year, float) and np.isnan(year)) or pd.isna(year):
        return vec
    y = int(year)
    idx = 0
    for i, e in enumerate(edges):
        if y <= e:
            idx = i
            break
        idx = i + 1
    vec[idx] = 1.0
    return vec


def content_vector(row: pd.Series) -> np.ndarray:
    return np.concatenate([genres_to_vec(row.get("genres")), year_to_bucket(row.get("year"))])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recency_weight(year: float | int | None, alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    if year is None or (isinstance(year, float) and np.isnan(year)) or pd.isna(year):
        return 1.0
    y = int(year)
    y = max(1980, min(2025, y))
    base = 0.6 + 0.4 * ((y - 1980) / (2025 - 1980))
    return (1 - alpha) * 1.0 + alpha * base
