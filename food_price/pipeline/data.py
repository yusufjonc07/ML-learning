"""Load every data source and merge it into one clean monthly panel.

Merge keys, by source:
    prices      -> the target (product, district, price) at (district, date)
    fuel        -> on "date"              (city-wide fuel grades)
    currency    -> on "date"              (USD/UZS)
    weather     -> on "date"              (national, production-weighted)
    population  -> on ["district", "date"] (per-district)

Generated caches (weather/population/fuel/currency CSVs) are read if present and
built on demand otherwise; a source that cannot be built is skipped with a
warning so the pipeline always runs on whatever is available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    """Read prices.csv and attach a month-start `date`; keep the modelling columns."""
    df = pd.read_csv(config.PRICES_CSV)
    df.columns = [c.lstrip("﻿") for c in df.columns]     # strip BOM on 'year'
    tokens = df["month"].astype(str).str.strip().str.lower()
    df["month_num"] = tokens.map(config.MONTH_MAP)
    if df["month_num"].isna().any():
        bad = sorted(tokens[df["month_num"].isna()].unique())
        raise ValueError(f"Unmapped month tokens in prices.csv: {bad} (add them to config.MONTH_MAP)")
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month_num"], day=1))
    return df[[config.DATE_COL, "product", "district", config.TARGET]].copy()


# ---------------------------------------------------------------------------
# Exogenous sources (read cache, else build, else skip)
# ---------------------------------------------------------------------------
def _load_or_build(path: Path, builder, label: str) -> pd.DataFrame | None:
    if Path(path).exists():
        return pd.read_csv(path, parse_dates=[config.DATE_COL])
    try:
        frame = builder()
    except Exception as exc:  # noqa: BLE001 - a missing source must not kill the run
        print(f"[data] WARNING: could not build {label} ({exc}); skipping.")
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[data] built + cached {label} -> {path.name}")
    return frame


def load_fuel() -> pd.DataFrame | None:
    def _build():
        from utils.fuel import load_fuel_rates
        return load_fuel_rates()
    return _load_or_build(config.FUEL_CSV, _build, "fuel")


def load_currency() -> pd.DataFrame | None:
    def _build():
        from utils.currency import load_currency_rates
        return load_currency_rates(cbu_dir=config.CBU_DIR)[["date", "USD"]]
    frame = _load_or_build(config.CURRENCY_CSV, _build, "currency")
    return frame[["date", "USD"]] if frame is not None and "USD" in frame else frame


def load_weather() -> pd.DataFrame | None:
    def _build():
        from utils.weather import download_weather
        return download_weather(out_path=None)
    return _load_or_build(config.WEATHER_CSV, _build, "weather")


def load_population() -> pd.DataFrame | None:
    def _build():
        from utils.population import build_population
        return build_population(out_path=None)
    return _load_or_build(config.POPULATION_CSV, _build, "population")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def build_panel(verbose: bool = True) -> pd.DataFrame:
    """Merge prices with every available exogenous source into one panel."""
    df = load_prices()
    if verbose:
        print(f"[data] prices: {len(df)} rows, {df['date'].dt.strftime('%Y-%m').nunique()} months, "
              f"{df['product'].nunique()} products x {df['district'].nunique()} districts")

    sources = [
        ("fuel", load_fuel(), ["date"]),
        ("currency", load_currency(), ["date"]),
        ("weather", load_weather(), ["date"]),
        ("population", load_population(), ["district", "date"]),
    ]
    for name, frame, keys in sources:
        if frame is None:
            if verbose:
                print(f"[data] - {name}: unavailable, skipped")
            continue
        before = df.shape[1]
        df = df.merge(frame, on=keys, how="left")
        if verbose:
            probe = next(c for c in frame.columns if c not in keys)
            print(f"[data] + {name}: +{df.shape[1] - before} cols, "
                  f"{df[probe].notna().mean():.0%} row coverage (merge on {keys})")
    return df
