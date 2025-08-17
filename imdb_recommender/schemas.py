
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

IMDB_CONST_RE = re.compile(r"^tt\d{7,}$")

class RatingRow(BaseModel):
    imdb_const: str = Field(...)
    my_rating: int = Field(..., ge=1, le=10)
    rated_at: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    genres: Optional[str] = None
    imdb_rating: Optional[float] = None
    num_votes: Optional[int] = None

    @field_validator("imdb_const")
    @classmethod
    def _valid_const(cls, v: str) -> str:
        if not IMDB_CONST_RE.match(v):
            raise ValueError(f"Invalid IMDb constant: {v}")
        return v

class WatchlistRow(BaseModel):
    imdb_const: str
    in_watchlist: bool = True
    title: Optional[str] = None
    year: Optional[int] = None
    genres: Optional[str] = None
    imdb_rating: Optional[float] = None
    num_votes: Optional[int] = None

    @field_validator("imdb_const")
    @classmethod
    def _valid_const(cls, v: str) -> str:
        if not IMDB_CONST_RE.match(v):
            raise ValueError(f"Invalid IMDb constant: {v}")
        return v

class Recommendation(BaseModel):
    imdb_const: str
    title: Optional[str] = None
    year: Optional[int] = None
    genres: Optional[str] = None
    score: float
    why_explainer: str

class ActionLogRow(BaseModel):
    timestamp_iso: str
    imdb_const: str
    action: str
    rating: Optional[int] = None
    notes: Optional[str] = None
    source: str = "cli"
    batch_id: str

    @field_validator("imdb_const")
    @classmethod
    def _valid_const(cls, v: str) -> str:
        if not IMDB_CONST_RE.match(v):
            raise ValueError(f"Invalid IMDb constant: {v}")
        return v
