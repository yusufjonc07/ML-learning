"""National (production-weighted) weather enrichment for the Tashkent food-price panel.

Why national, not per-district
------------------------------
The price panel is *retail* food prices in Tashkent city (the ``district`` column
is the 12 tumanlar of Tashkent + the city-wide "Shahar bo'yicha"). But Tashkent is
a consumption hub, not a growing one: the weather that actually moves the price of
`Kartoshka`, `Piyoz`, `Pomidor`, `Olma`, grain and dairy is the weather in the
farming regions - the Fergana Valley, Samarkand, Kashkadarya, Tashkent region, etc.
And every Tashkent district falls in a single ERA5 grid cell, so a per-district split
just produced 13 identical copies of an irrelevant signal.

So this module builds ONE weather record per month, as a *production-weighted average*
over Uzbekistan's agricultural regions (deserts and the city weight ~0). It merges on
``date`` alone (a month-start Timestamp, same convention as ``currency.py``); pandas
broadcasts the national row across every district/product row of that month::

    df = df.merge(weather, on="date", how="left")

Weather acts on prices with a lag (growing season -> harvest -> market), so the module
also downloads a warm-up window before the panel start and emits lagged copies of the
key drivers, so July-2023 rows get populated lags/rolls instead of NaN.

Data source: https://open-meteo.com/en/docs/historical-weather-api  (free, key-less).
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; guard just in case
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover - extremely unlikely
    Retry = None

try:  # progress bars are nice-to-have, not a hard dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fall back to a no-op wrapper
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []


# ---------------------------------------------------------------------------
# 1. Agricultural regions of Uzbekistan + their output weights
# ---------------------------------------------------------------------------
# Representative point per region (regional centre / farming belt). Tashkent *city*
# is intentionally excluded - it grows almost nothing. Coordinates need only be
# accurate to the ERA5 grid (~10-25 km), so regional capitals are fine.
REGION_COORDS: dict[str, tuple[float, float]] = {
    "Andijan":        (40.7821, 72.3442),
    "Bukhara":        (39.7675, 64.4231),
    "Fergana":        (40.3864, 71.7864),
    "Jizzakh":        (40.1158, 67.8422),
    "Kashkadarya":    (38.8600, 65.7887),   # Qarshi
    "Khorezm":        (41.5500, 60.6333),   # Urgench
    "Namangan":       (40.9983, 71.6726),
    "Navoi":          (40.0844, 65.3792),
    "Samarkand":      (39.6542, 66.9597),
    "Surkhandarya":   (37.2242, 67.2783),   # Termez
    "Syrdarya":       (40.4897, 68.7842),   # Gulistan
    "Tashkent region": (41.0000, 69.4000),  # farming belt around (not in) the city
    "Karakalpakstan": (42.4531, 59.6103),   # Nukus
}

# APPROXIMATE share of gross agricultural output per region. These are rough,
# hand-set priors (deserts/mining regions low, irrigated valleys high) - REPLACE
# with official figures from stat.uz (gross agricultural output by region) when you
# have them. They do not need to sum to 1; they are normalised at runtime.
REGION_WEIGHTS: dict[str, float] = {
    "Samarkand":       11.0,
    "Tashkent region": 10.0,
    "Kashkadarya":      9.0,
    "Fergana":          9.0,
    "Andijan":          8.5,
    "Namangan":         7.5,
    "Surkhandarya":     7.0,
    "Bukhara":          6.0,
    "Khorezm":          5.5,
    "Jizzakh":          5.0,
    "Karakalpakstan":   4.5,
    "Syrdarya":         3.5,
    "Navoi":            3.0,   # mostly desert -> low
}

# ---------------------------------------------------------------------------
# 2. API configuration
# ---------------------------------------------------------------------------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily aggregates the archive exposes directly (ERA5). Units returned:
#   temps °C, precip mm, wind km/h, radiation MJ/m², et0 mm, sunshine s.
# Only variables we actually consume are requested (Open-Meteo's free-tier call
# cost scales with variable count, so we do not fetch anything we discard).
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
]

# Hourly fields the API has NO daily aggregate for; we roll them up ourselves.
HOURLY_VARS = [
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_temperature_0_to_7cm",
]

# Thresholds (documented so they are easy to tune). If you retune HOT_DAY_C, note
# that the `days_above_35` column name still encodes 35 - rename it to match.
RAIN_DAY_MM = 1.0      # WMO "wet day": >= 1 mm of precipitation
HOT_DAY_C = 35.0       # heat-stress threshold; used by days_above_35 AND heatwave (both >=)
HEATWAVE_RUN = 3       # a heatwave = this many consecutive hot days
GDD_BASE_C = 10.0      # growing-degree-day base temperature
COMPLETE_MONTH_DAYS = 28  # a month with fewer real days is treated as partial

DEFAULT_START = "2021-01-01"   # first month of the actual price panel (prices.csv runs 2021-01..2026-02)
WARMUP_MONTHS = 6              # extra history so the first panel month has lags/rolls

# Features that get explicit lagged copies (the harvest -> price mechanism).
LAG_FEATURES = ["temp_mean", "precipitation_sum", "et0", "drought_index", "gdd", "soil_moisture_0_7"]
WEATHER_LAGS = (1, 2, 3, 6)


# ---------------------------------------------------------------------------
# 3. Robust download layer
# ---------------------------------------------------------------------------
def _build_session(total_retries: int = 4, backoff: float = 1.5) -> requests.Session:
    """A session that retries transient network/HTTP failures automatically.

    The adapter transparently retries connection resets and 429/5xx with
    exponential backoff and honours ``Retry-After``; JSON-level errors and the
    out-of-range 400 are handled in `_fetch`.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "food-price-weather/1.0"})
    if Retry is not None:
        retry = Retry(
            total=total_retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    return session


def _fetch(session, lat, lon, start, end, *, timeout=60, max_tries=3, pause=1.0):
    """Fetch one point's daily+hourly archive.

    The adapter already retries transient transport/5xx/429 failures, so this loop
    only adds: (1) a clean, non-retried handling of the archive's out-of-range 400
    (it clamps ``end`` to the API's advertised maximum date and retries once - this
    is exactly what happens when the caller's clock is a few hours ahead of the
    ERA5 boundary), and (2) a small retry for the API's JSON-level ``error`` payloads.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",          # aligns daily min/max to local calendar days
    }
    last_err = "unknown error"
    for attempt in range(1, max_tries + 1):
        try:
            resp = session.get(ARCHIVE_URL, params=params, timeout=timeout)

            if resp.status_code == 400:
                # e.g. "... out of allowed range from 1940-01-01 to 2026-07-10"
                reason = ""
                try:
                    reason = resp.json().get("reason", "")
                except Exception:
                    reason = resp.text[:200]
                m = re.search(r"to (\d{4}-\d{2}-\d{2})", reason)
                if m and params["end_date"] > m.group(1):
                    params["end_date"] = m.group(1)   # clamp to what the archive has
                    continue                          # retry once, not counted as failure
                raise RuntimeError(f"HTTP 400: {reason}")  # any other 400 is non-retryable

            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(payload.get("reason", "unknown Open-Meteo error"))
            if not payload.get("daily", {}).get("time") or not payload.get("hourly", {}).get("time"):
                raise RuntimeError("empty daily/hourly block for the requested range")
            return payload
        except Exception as exc:  # noqa: BLE001 - retry JSON/transient errors
            last_err = exc
            if attempt < max_tries:
                time.sleep(pause * attempt)
    raise RuntimeError(f"Open-Meteo failed for ({lat},{lon}) after {max_tries} tries: {last_err}")


# ---------------------------------------------------------------------------
# 4. Reshape raw JSON -> monthly features for ONE point
# ---------------------------------------------------------------------------
def _month_key(times: pd.Series) -> pd.Series:
    return pd.to_datetime(times).dt.to_period("M").dt.to_timestamp()  # month-start


def _heatwave_days_per_month(daily: pd.DataFrame) -> pd.Series:
    """Days belonging to a run of >= HEATWAVE_RUN consecutive hot days (temp_max >= HOT_DAY_C).

    Runs are found on the full sorted series so a heatwave straddling a month
    boundary counts its days in whichever month each falls. Returns a month-indexed
    Series (empty if there are no days).
    """
    s = daily.sort_values("time")
    hot = (s["temperature_2m_max"] >= HOT_DAY_C).to_numpy()
    if len(hot) == 0:
        return pd.Series(dtype="int64")
    # Label consecutive equal-value runs; keep only hot runs that reach the length.
    run_id = np.cumsum(np.concatenate(([True], hot[1:] != hot[:-1])))
    in_heatwave = np.zeros(len(hot), dtype=bool)
    for rid in np.unique(run_id):
        idx = run_id == rid
        if hot[idx][0] and idx.sum() >= HEATWAVE_RUN:
            in_heatwave[idx] = True
    return pd.Series(in_heatwave, index=s["month"].to_numpy()).groupby(level=0).sum()


def _aggregate_point(payload: dict) -> pd.DataFrame:
    """Collapse one region's daily+hourly data into monthly base features (no derived)."""
    daily = pd.DataFrame(payload["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily["month"] = _month_key(daily["time"])
    hourly = pd.DataFrame(payload["hourly"])
    hourly["month"] = _month_key(hourly["time"])

    # Daily growing-degree-days (base 10 °C, negatives clipped) and diurnal range.
    daily["gdd"] = np.maximum(0.0, (daily["temperature_2m_max"] + daily["temperature_2m_min"]) / 2 - GDD_BASE_C)
    daily["dtr"] = daily["temperature_2m_max"] - daily["temperature_2m_min"]

    dg = daily.groupby("month")
    monthly = pd.DataFrame({
        "temp_mean":         dg["temperature_2m_mean"].mean(),
        "temp_max":          dg["temperature_2m_max"].max(),
        "temp_min":          dg["temperature_2m_min"].min(),
        "temp_diurnal_range": dg["dtr"].mean(),                             # mean daily max-min
        # 'rain' columns are precipitation (rain + snow water-equivalent).
        "precipitation_sum": dg["precipitation_sum"].sum(),
        "rain_days":         dg["precipitation_sum"].agg(lambda s: float((s >= RAIN_DAY_MM).sum())),
        "max_daily_rain":    dg["precipitation_sum"].max(),
        "wind_max":          dg["wind_speed_10m_max"].max(),
        "wind_gust_max":     dg["wind_gusts_10m_max"].max(),
        "solar_radiation":   dg["shortwave_radiation_sum"].sum(),          # MJ/m² total for month
        "sunshine_duration": dg["sunshine_duration"].sum() / 3600.0,       # seconds -> hours
        "et0":               dg["et0_fao_evapotranspiration"].sum(),       # mm total for month
        "gdd":               dg["gdd"].sum(),
        "days_above_35":     dg["temperature_2m_max"].agg(lambda s: float((s >= HOT_DAY_C).sum())),
        "frost_days":        dg["temperature_2m_min"].agg(lambda s: float((s < 0).sum())),
        # n_days counts days with real data (non-null), so partial months are honest.
        "n_days":            dg["temperature_2m_mean"].count(),
    })
    monthly["temp_range"] = monthly["temp_max"] - monthly["temp_min"]   # extreme monthly spread
    monthly["days_below_0"] = monthly["frost_days"]                     # alias requested in the brief
    monthly["heatwave_days"] = (
        _heatwave_days_per_month(daily).reindex(monthly.index).fillna(0).astype(float)
    )

    hg = hourly.groupby("month")
    monthly["humidity_mean"]      = hg["relative_humidity_2m"].mean()
    monthly["humidity_max"]       = hg["relative_humidity_2m"].max()
    monthly["wind_mean"]          = hg["wind_speed_10m"].mean()
    monthly["surface_pressure"]   = hg["surface_pressure"].mean()
    monthly["soil_moisture_0_7"]  = hg["soil_moisture_0_to_7cm"].mean()
    monthly["soil_moisture_7_28"] = hg["soil_moisture_7_to_28cm"].mean()
    monthly["soil_temperature"]   = hg["soil_temperature_0_to_7cm"].mean()

    return monthly.reset_index(names="date")


# Base features produced per region (everything except the cross-month derived cols).
_BASE_FEATURES = [
    "temp_mean", "temp_max", "temp_min", "temp_range", "temp_diurnal_range",
    "precipitation_sum", "rain_days", "max_daily_rain",
    "humidity_mean", "humidity_max",
    "wind_mean", "wind_max", "wind_gust_max",
    "solar_radiation", "sunshine_duration", "et0",
    "soil_moisture_0_7", "soil_moisture_7_28", "soil_temperature", "surface_pressure",
    "heatwave_days", "frost_days", "days_above_35", "days_below_0", "gdd",
]


# ---------------------------------------------------------------------------
# 5. Combine regions -> one production-weighted national series
# ---------------------------------------------------------------------------
def _weighted_national(per_region: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Production-weighted average of every base feature, per month.

    NaN-safe: for each feature/month the weights are renormalised over the regions
    that actually reported a value, so a region that failed to download (or has a
    gap) shifts the average slightly instead of poisoning it with NaN.
    """
    long = pd.concat(
        [df.assign(region=name) for name, df in per_region.items()],
        ignore_index=True,
    )
    long["w"] = long["region"].map(weights).astype(float)

    out = {}
    for col in _BASE_FEATURES:
        sub = long[["date", "w", col]].dropna(subset=[col])
        num = (sub[col] * sub["w"]).groupby(sub["date"]).sum()
        den = sub["w"].groupby(sub["date"]).sum()
        out[col] = num / den
    national = pd.DataFrame(out)
    national["n_days"] = long.groupby("date")["n_days"].max()   # coverage, not weighted
    return national.sort_index().reset_index(names="date")


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Water balance, rolling windows, seasonal anomaly, cumulative GDD - on the national series."""
    df = df.sort_values("date").reset_index(drop=True).copy()

    # Water balance: rainfall minus reference ET (mm). Negative => drought pressure.
    df["drought_index"] = df["precipitation_sum"] - df["et0"]

    # Rolling means (min_periods=1 keeps early months usable; weather is exogenous,
    # so including the current month is fine). Use .sum() instead for accumulation.
    df["rainfall_roll3"]    = df["precipitation_sum"].rolling(3, min_periods=1).mean()
    df["rainfall_roll6"]    = df["precipitation_sum"].rolling(6, min_periods=1).mean()
    df["temperature_roll3"] = df["temp_mean"].rolling(3, min_periods=1).mean()
    df["temperature_roll6"] = df["temp_mean"].rolling(6, min_periods=1).mean()

    # Rainfall anomaly vs this month's climatology. The mean is taken over COMPLETE
    # months only, so a partial current month cannot bias the normal (history is
    # short - 2023+ - so treat this as a coarse seasonal signal, not a 30-yr normal).
    moy = df["date"].dt.month
    complete = df["n_days"] >= COMPLETE_MONTH_DAYS
    clim = (
        df[complete].groupby(df.loc[complete, "date"].dt.month)["precipitation_sum"].mean()
    )
    df["rainfall_anomaly"] = df["precipitation_sum"] - moy.map(clim)

    # Cumulative GDD within each calendar year (heat accumulated so far).
    df["gdd_cumulative"] = df.groupby(df["date"].dt.year)["gdd"].cumsum()
    return df


def _add_weather_lags(df: pd.DataFrame, features=LAG_FEATURES, lags=WEATHER_LAGS) -> pd.DataFrame:
    """Shifted copies of the causal drivers (single national series, sorted by date)."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    for feat in features:
        for k in lags:
            df[f"{feat}_lag_{k}"] = df[feat].shift(k)
    return df


def _lag_columns(features=LAG_FEATURES, lags=WEATHER_LAGS) -> list[str]:
    return [f"{f}_lag_{k}" for f in features for k in lags]


# Final column order: requested schema, then the useful extras, then the lags.
_OUTPUT_COLUMNS = [
    "date",
    "temp_mean", "temp_max", "temp_min", "temp_range", "temp_diurnal_range",
    "precipitation_sum", "rain_days", "max_daily_rain",
    "humidity_mean", "humidity_max",
    "wind_mean", "wind_max", "wind_gust_max",
    "solar_radiation", "sunshine_duration", "et0",
    "soil_moisture_0_7", "soil_moisture_7_28", "soil_temperature", "surface_pressure",
    "heatwave_days", "frost_days", "days_above_35", "days_below_0",
    "gdd", "gdd_cumulative", "drought_index", "rainfall_anomaly",
    "rainfall_roll3", "rainfall_roll6", "temperature_roll3", "temperature_roll6",
    "n_days",
] + _lag_columns()


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------
def download_weather(
    regions: dict[str, tuple[float, float]] = REGION_COORDS,
    weights: dict[str, float] = REGION_WEIGHTS,
    start: str = DEFAULT_START,
    end: str | None = None,
    out_path: str | Path | None = "weather_monthly.csv",
    *,
    warmup_months: int = WARMUP_MONTHS,
    pause: float = 1.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Build the monthly, production-weighted national weather table.

    Parameters
    ----------
    regions : region name -> (lat, lon).
    weights : region name -> agricultural-output weight (need not sum to 1).
    start   : first month to KEEP (ISO ``YYYY-MM-DD``); the panel start.
    end     : last day to download; defaults to today (UTC). The archive lags a
              few days, so the current month may be partial - the ``n_days`` column
              flags it (``n_days < 28``).
    warmup_months : extra history downloaded before ``start`` so the first kept
              month already has populated lag/rolling features.
    out_path : CSV destination (``None`` to skip writing).

    Returns
    -------
    One row per month, mergeable via ``df.merge(weather, on="date", how="left")``.
    """
    if end is None:
        # UTC (not local) date: the archive's max end_date is a UTC boundary, and a
        # caller a few hours ahead of UTC (KST/Tashkent) must not request tomorrow.
        end = datetime.now(timezone.utc).date().isoformat()

    panel_start = pd.Timestamp(start)
    dl_start = (panel_start - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")

    session = session or _build_session()
    per_region: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    for name, (lat, lon) in tqdm(list(regions.items()), desc="regions", unit="region"):
        try:
            payload = _fetch(session, lat, lon, dl_start, end, pause=pause)
            per_region[name] = _aggregate_point(payload)
            time.sleep(pause)  # polite spacing on the free tier
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failures.append(f"{name}: {exc}")

    if not per_region:
        raise RuntimeError("Every region failed to download:\n  " + "\n  ".join(failures))
    if failures:
        used_w = sum(weights.get(r, 0) for r in per_region)
        total_w = sum(weights.values())
        print(
            f"[weather] WARNING - {len(failures)} region(s) failed; national average "
            f"built from {used_w / total_w:.0%} of the agricultural weight:\n  "
            + "\n  ".join(failures)
        )

    national = _weighted_national(per_region, weights)
    national = _add_derived(national)
    national = _add_weather_lags(national)
    national = national[national["date"] >= panel_start]          # drop warm-up months
    national = national.reindex(columns=_OUTPUT_COLUMNS).sort_values("date").reset_index(drop=True)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        national.to_csv(out_path, index=False)
        print(f"[weather] wrote {len(national)} months x {national.shape[1]} cols -> {out_path}")
    return national


def main() -> None:
    weather = download_weather()
    print(weather[["date", "temp_mean", "precipitation_sum", "et0", "drought_index", "n_days"]].to_string(index=False))


if __name__ == "__main__":
    main()


# Example usage (in FoodTashkent.ipynb, after the panel's month-start `date` exists):
#     from utils.weather import download_weather
#     weather = download_weather(end="2026-06-30")     # cached to weather_monthly.csv
#     df = df.merge(weather, on="date", how="left")    # merge on date ONLY (national signal)
#
# Reloading the CSV later? Parse the date so the merge key stays datetime:
#     weather = pd.read_csv("weather_monthly.csv", parse_dates=["date"])
