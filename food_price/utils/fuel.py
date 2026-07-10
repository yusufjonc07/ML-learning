"""Monthly fuel prices for Tashkent (mirrors currency.py's load-and-tidy pattern).

`fuel_scrap.py` scrapes the raw per-station price tables from goldenpages.uz;
this module turns them into one clean row per month (all stations averaged),
reindexed onto a continuous month-start range with gaps interpolated - exactly
the AI-80 / AI-92 IMPORT / AI-95 / AI-98 / AI-100 / Diesel series the model uses.
"""

from __future__ import annotations

import pandas as pd

from utils.fuel_scrap import scrape_fuel_year

# Columns that are not fuel prices, dropped before averaging. "AI-92 UZB" is
# empty for several years, so it is excluded rather than carried as mostly-NaN.
_NON_PRICE = ["Gas station name", "AI-92 UZB"]


def load_fuel_rates(years=range(2021, 2027), freq: str = "MS") -> pd.DataFrame:
    """Scrape every year in `years` and return monthly average fuel prices.

    Returns one row per month (month-start dates) with a numeric column per fuel
    grade, ready to merge onto the panel with ``on="date"``.
    """
    fuel = pd.concat([scrape_fuel_year(y) for y in years], ignore_index=True)

    # Prices arrive as strings ("12 800", "-" for missing) -> numbers.
    price_cols = [c for c in fuel.columns if c not in ("Gas station name", "date")]
    for c in price_cols:
        fuel[c] = pd.to_numeric(
            fuel[c].astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", pd.NA),
            errors="coerce",
        )

    drop = [c for c in _NON_PRICE if c in fuel.columns]
    monthly = (
        fuel.drop(columns=drop)
        .assign(date=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date")
        .mean(numeric_only=True)
        .round(0)
        .reset_index()
    )

    # Reindex to a gap-free monthly grid and fill the (few) missing months.
    full = pd.date_range(monthly["date"].min(), monthly["date"].max(), freq=freq)
    monthly = (
        monthly.set_index("date").reindex(full).rename_axis("date")
        .interpolate(method="linear").bfill().ffill().reset_index()
    )
    return monthly
