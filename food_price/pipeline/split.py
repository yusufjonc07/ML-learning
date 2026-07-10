"""Time-based train/test split.

Forecasting must never shuffle time: the test set is the most recent months, so
the model is only ever asked to predict the future from the past. Everything
before SPLIT_DATE trains; everything on/after it is held out.
"""

from __future__ import annotations

import pandas as pd

import config


def time_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, split_date: str | None = None):
    """Split by date into (X_train, X_test, y_train, y_test).

    `dates` must be aligned to X/y (same index). Returns the four frames; recover
    the test dates for plotting with ``dates[dates >= split_date]``.
    """
    split = pd.Timestamp(split_date or config.SPLIT_DATE)
    train = dates < split
    test = dates >= split
    return X[train], X[test], y[train], y[test]
