"""Turn the merged panel into a numeric modelling matrix.

Adds the price-history features (lags + rolling mean/std, computed *within* each
product-district series so no future price leaks in), calendar/seasonal features,
and integer codes for the categoricals; then selects the numeric feature columns
and drops the rows whose lags are still undefined (each series' first months).
"""

from __future__ import annotations

import pandas as pd

import config
from utils.featuring import create_lags, create_rolling_features, create_calendar_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lags, rolling stats, calendar features and categorical codes."""
    df = create_lags(df, col=config.TARGET, lags=config.LAGS)
    df = create_rolling_features(df, col=config.TARGET, windows=config.ROLL_WINDOWS)
    df = create_calendar_features(df, date_col=config.DATE_COL)
    for col in config.GROUP_KEYS:                       # product, district -> codes
        df[f"{col}_code"] = df[col].astype("category").cat.codes
    return df


def assemble(df: pd.DataFrame):
    """Return (X, y, dates): numeric features, target, and the date of each row.

    X excludes the target, the raw string categoricals (object dtype), the date,
    and the identifier/flag columns in config.DROP_FEATURES. Rows with any missing
    feature (series starts, where lags are undefined) are dropped jointly.
    """
    feat = build_features(df)
    y = feat[config.TARGET]
    dates = feat[config.DATE_COL]

    X = feat.select_dtypes("number").drop(columns=[config.TARGET], errors="ignore")
    X = X.drop(columns=[c for c in config.DROP_FEATURES if c in X.columns], errors="ignore")

    keep = X.notna().all(axis=1) & y.notna()
    return (
        X[keep].reset_index(drop=True),
        y[keep].reset_index(drop=True),
        dates[keep].reset_index(drop=True),
    )
