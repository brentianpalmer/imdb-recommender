from __future__ import annotations

import pandas as pd

MOVIE_TYPES = {"movie", "short"}
TV_TYPES = {"tvSeries", "tvMiniSeries", "tvMovie", "tvSpecial"}


def filter_by_content_type(df: pd.DataFrame, content_type: str) -> pd.DataFrame:
    """Filter a recommendations dataframe by content type.

    Parameters
    ----------
    df:
        DataFrame containing a ``titleType`` column.
    content_type:
        One of ``{"all", "movies", "tv"}``.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Raises
    ------
    ValueError
        If ``content_type`` is not recognised or the required column is
        missing when a filter other than ``"all"`` is requested.
    """
    if content_type == "all":
        return df

    if "titleType" not in df.columns:
        raise ValueError("titleType metadata required for content-type filtering.")

    if content_type == "movies":
        return df[df["titleType"].isin(MOVIE_TYPES)]
    if content_type == "tv":
        return df[df["titleType"].isin(TV_TYPES)]

    raise ValueError(f"Unknown content_type: {content_type}")
