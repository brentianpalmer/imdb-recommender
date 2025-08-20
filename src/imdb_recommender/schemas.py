from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

IMDB_CONST_RE = re.compile(r"^tt\d{7,}$")


class RatingRow(BaseModel):
    imdb_const: str = Field(...)
    my_rating: int = Field(..., ge=1, le=10)
    rated_at: str | None = None
    title: str | None = None
    year: int | None = None
    genres: str | None = None
    imdb_rating: float | None = None
    num_votes: int | None = None

    @field_validator("imdb_const")
    @classmethod
    def _valid_const(cls, v: str) -> str:
        if not IMDB_CONST_RE.match(v):
            raise ValueError(f"Invalid IMDb constant: {v}")
        return v


class WatchlistRow(BaseModel):
    imdb_const: str
    in_watchlist: bool = True
    title: str | None = None
    year: int | None = None
    genres: str | None = None
    imdb_rating: float | None = None
    num_votes: int | None = None

    @field_validator("imdb_const")
    @classmethod
    def _valid_const(cls, v: str) -> str:
        if not IMDB_CONST_RE.match(v):
            raise ValueError(f"Invalid IMDb constant: {v}")
        return v


class Recommendation(BaseModel):
    imdb_const: str
    title: str | None = None
    year: int | None = None
    genres: str | None = None
    score: float
    why_explainer: str


class ActionLogRow(BaseModel):
    timestamp_iso: str
    imdb_const: str
    action: str
    rating: int | None = None
    notes: str | None = None
    source: str = "cli"
    batch_id: str

    @field_validator("imdb_const")
    @classmethod
    def _valid_const(cls, v: str) -> str:
        if not IMDB_CONST_RE.match(v):
            raise ValueError(f"Invalid IMDb constant: {v}")
        return v
