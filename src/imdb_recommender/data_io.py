from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


@dataclass
class Dataset:
    ratings: pd.DataFrame
    watchlist: pd.DataFrame

    @property
    def catalog(self) -> pd.DataFrame:
        r = self.ratings.set_index("imdb_const", drop=False)
        w = self.watchlist.set_index("imdb_const", drop=False)
        out = r.combine_first(w).reset_index(drop=True)
        for col in ["title", "year", "genres", "imdb_rating", "num_votes", "title_type"]:
            if col not in out.columns:
                out[col] = np.nan
        return out


@dataclass
class IngestResult:
    dataset: Dataset
    warnings: list[str]


def load_ratings_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = _norm_cols(df)
    const_col = _pick(df, ["const", "imdb_const", "title_id", "titleid"])
    yr_col = _pick(df, ["your_rating", "my_rating", "rating"])
    date_col = _pick(df, ["date_rated", "created", "timestamp"])
    title_col = _pick(df, ["title"])
    year_col = _pick(df, ["year"])
    genres_col = _pick(df, ["genres"])
    imdb_rating_col = _pick(df, ["imdb_rating", "average_rating"])
    votes_col = _pick(df, ["num_votes", "votes"])
    title_type_col = _pick(df, ["title_type"])
    if not const_col or not yr_col:
        raise ValueError("Could not find required columns in ratings CSV")
    out = pd.DataFrame(
        {
            "imdb_const": df[const_col].astype(str),
            "my_rating": df[yr_col].astype(float).round().astype("Int64"),
        }
    )
    out["rated_at"] = df[date_col].astype(str) if date_col else None
    out["title"] = df[title_col].astype(str) if title_col else None
    out["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64") if year_col else None
    out["genres"] = df[genres_col].astype(str) if genres_col else None
    out["imdb_rating"] = (
        pd.to_numeric(df[imdb_rating_col], errors="coerce") if imdb_rating_col else np.nan
    )
    out["num_votes"] = (
        pd.to_numeric(df[votes_col], errors="coerce").astype("Int64") if votes_col else None
    )
    out["title_type"] = df[title_type_col].astype(str) if title_type_col else None
    out = out.dropna(subset=["imdb_const"])
    if "rated_at" in out.columns and out["rated_at"].notna().any():
        out = out.sort_values("rated_at").drop_duplicates("imdb_const", keep="last")
    else:
        out = out.drop_duplicates("imdb_const", keep="last")
    out["imdb_const"] = out["imdb_const"].astype(str)
    out["my_rating"] = out["my_rating"].astype("Int64")
    return out.reset_index(drop=True)


def load_watchlist(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8")
    df = _norm_cols(df)
    const_col = _pick(df, ["const", "imdb_const", "title_id", "titleid"])
    title_col = _pick(df, ["title"])
    year_col = _pick(df, ["year"])
    genres_col = _pick(df, ["genres"])
    imdb_rating_col = _pick(df, ["imdb_rating", "average_rating"])
    votes_col = _pick(df, ["num_votes", "votes"])
    title_type_col = _pick(df, ["title_type"])
    if not const_col:
        raise ValueError("Could not find 'const' column in watchlist")
    out = pd.DataFrame({"imdb_const": df[const_col].astype(str), "in_watchlist": True})
    out["title"] = df[title_col].astype(str) if title_col else None
    out["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64") if year_col else None
    out["genres"] = df[genres_col].astype(str) if genres_col else None
    out["imdb_rating"] = (
        pd.to_numeric(df[imdb_rating_col], errors="coerce") if imdb_rating_col else np.nan
    )
    out["num_votes"] = (
        pd.to_numeric(df[votes_col], errors="coerce").astype("Int64") if votes_col else None
    )
    out["title_type"] = df[title_type_col].astype(str) if title_type_col else None
    out = out.dropna(subset=["imdb_const"]).drop_duplicates("imdb_const", keep="last")
    out["imdb_const"] = out["imdb_const"].astype(str)
    return out.reset_index(drop=True)


def ingest_sources(ratings_csv: str, watchlist_path: str, data_dir: str = "data") -> IngestResult:
    r = load_ratings_csv(ratings_csv)
    w = load_watchlist(watchlist_path)
    warnings = []
    ds = Dataset(ratings=r, watchlist=w)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    try:
        r.to_parquet(Path(data_dir) / "ratings_normalized.parquet", index=False)
        w.to_parquet(Path(data_dir) / "watchlist_normalized.parquet", index=False)
    except Exception:
        r.to_csv(Path(data_dir) / "ratings_normalized.csv", index=False)
        w.to_csv(Path(data_dir) / "watchlist_normalized.csv", index=False)
    return IngestResult(dataset=ds, warnings=warnings)
