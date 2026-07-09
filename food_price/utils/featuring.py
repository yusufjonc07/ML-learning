import numpy as np
import pandas as pd

# Feature engineering for the Tashkent food-price panel.
#
# The data is not a single time series: each (product, district) pair is its
# own monthly series. So every time-based feature is computed *within* that
# group after sorting by date - otherwise a lag/rolling value would leak the
# previous product's or district's price into the current row.
GROUP_KEYS = ["product", "district"]


def create_lags(df, col="price", lags=(1, 2, 3, 12), group_keys=GROUP_KEYS):
    """Add the value of `col` from previous months as new columns.

    Lags are in months: 1 = last month, 12 = same month one year ago.
    Each (product, district) series is shifted independently, so the first
    `lag` rows of every series come out as NaN (no earlier data to borrow).
    """
    df = df.sort_values(group_keys + ["date"]).copy()
    grouped = df.groupby(group_keys)[col]
    for lag in lags:
        df[f"{col}_lag_{lag}"] = grouped.shift(lag)
    return df


def create_rolling_features(df, col="price", windows=(3, 6, 12), group_keys=GROUP_KEYS):
    """Add rolling mean/std of `col` over trailing windows (in months).

    The series is shifted by one month first, so each row's window only
    covers *past* months - including the current price would leak the target.
    min_periods=1 keeps early rows usable instead of dropping them to NaN.
    """
    df = df.sort_values(group_keys + ["date"]).copy()
    past = df.groupby(group_keys)[col].shift(1)          # only look backwards
    by_group = past.groupby([df[k] for k in group_keys])  # regroup the shifted series
    for window in windows:
        df[f"{col}_roll_mean_{window}"] = by_group.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std_{window}"] = by_group.transform(
            lambda s: s.rolling(window, min_periods=1).std()
        )
    return df


def create_calendar_features(df, date_col="date"):
    """Derive seasonality features from the date column.

    Food prices are seasonal, so we expose the month/quarter directly and add
    a cyclical (sin/cos) encoding so the model treats December and January as
    neighbours instead of the far-apart numbers 12 and 1.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["month_of_year"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df


# Example usage (in FoodTashkent.ipynb, after `date` and `price` exist):
#     from utils.featuring import (
#         create_lags, create_rolling_features, create_calendar_features,
#     )
#     df = create_lags(df)
#     df = create_rolling_features(df)
#     df = create_calendar_features(df)
