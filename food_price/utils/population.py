"""Monthly population per Tashkent district for the food-price panel.

Reality check
-------------
There is NO monthly district-level population series for Tashkent - official
figures (State Statistics Committee / Tashkent city statistics) are published
ANNUALLY, as the permanent population "as of 1 January". Population also moves
smoothly, not in monthly jumps. So the honest and standard approach - and what
this module does - is to take annual anchor values and **log-linearly interpolate
them to monthly** (linear growth in log space = constant %/month between anchors),
extrapolating the tail with the most recent annual growth rate.

Unlike weather (a city-wide signal), population genuinely differs across districts,
so this table merges on BOTH keys::

    df = df.merge(population, on=["district", "date"], how="left")

with ``date`` a month-start Timestamp (same convention as the other utils).

!! THE SEED NUMBERS BELOW ARE APPROXIMATE PLACEHOLDERS !!
Replace ``POP_2021`` / ``ANNUAL_GROWTH`` with official stat.uz district figures
(or drop a ``district,year,population`` CSV in and use ``load_annual_from_csv``).
As shipped, the series is a smooth demographic trend, not a measured count.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed data  (APPROXIMATE - replace with official figures)
# ---------------------------------------------------------------------------
# Keys MUST match the `district` strings in prices.csv exactly (ASCII apostrophes).
# Approximate permanent population as of 2021-01-01, persons. These are rough
# order-of-magnitude estimates for the 12 tumanlar of Tashkent city, NOT official
# counts - swap in real numbers before trusting any per-capita analysis.
POP_2021: dict[str, int] = {
    "Bektemir":        48_000,
    "Chilonzor":       275_000,
    "Mirobod":         175_000,
    "Mirzo Ulug'bek":  350_000,
    "Olmazor":         320_000,
    "Shayxontoxur":    295_000,
    "Sirg'ali":        240_000,
    "Uchtepa":         245_000,
    "Yakkasaroy":      118_000,
    "Yangihayot":      210_000,
    "Yashnobod":       270_000,
    "Yunusobod":       310_000,
}

# Assumed annual growth per district (Tashkent grows ~2%/yr; Yangihayot is a new,
# fast-growing district split off ~2020). Used to synthesise the yearly anchors.
DEFAULT_GROWTH = 0.020
ANNUAL_GROWTH: dict[str, float] = {name: DEFAULT_GROWTH for name in POP_2021}
ANNUAL_GROWTH["Yangihayot"] = 0.040
ANNUAL_GROWTH["Sirg'ali"] = 0.030   # also expanding (new residential belt)

# The city-wide row in prices.csv; its population is the sum of all districts.
CITYWIDE_KEY = "Shahar bo'yicha"

DEFAULT_START = "2021-01-01"   # first month of the price panel


# ---------------------------------------------------------------------------
# Annual anchors
# ---------------------------------------------------------------------------
def _seed_annual(years: range) -> dict[str, dict[int, float]]:
    """Synthesise yearly anchors from POP_2021 x growth, for each year in `years`.

    This is where the *modelled* (non-official) numbers come from. Providing real
    per-year values via `load_annual_from_csv` overrides all of this.
    """
    annual: dict[str, dict[int, float]] = {}
    for name, base in POP_2021.items():
        g = ANNUAL_GROWTH.get(name, DEFAULT_GROWTH)
        annual[name] = {y: base * (1 + g) ** (y - 2021) for y in years}
    return annual


def load_annual_from_csv(path: str | Path) -> dict[str, dict[int, float]]:
    """Load official anchors from a tidy ``district,year,population`` CSV.

    Any district/year you supply overrides the seed; districts you omit fall back
    to the synthesised seed inside `build_population`.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    d, y, p = cols["district"], cols["year"], cols["population"]
    annual: dict[str, dict[int, float]] = {}
    for _, row in df.iterrows():
        annual.setdefault(str(row[d]), {})[int(row[y])] = float(row[p])
    return annual


# ---------------------------------------------------------------------------
# Interpolation: annual anchors -> monthly series
# ---------------------------------------------------------------------------
def _interpolate_district(anchors: dict[int, float], months: pd.DatetimeIndex) -> np.ndarray:
    """Log-linear interpolation of one district's yearly anchors onto `months`.

    Interpolates in log space (so growth is a constant % between anchors) and
    extrapolates both ends with the nearest segment's slope, so months after the
    last official year keep growing at the last observed rate instead of flat-lining.
    """
    yrs = sorted(anchors)
    ax = np.array([pd.Timestamp(year=y, month=1, day=1).value for y in yrs], dtype=float)
    ay = np.log(np.array([anchors[y] for y in yrs], dtype=float))
    mx = months.values.astype("datetime64[ns]").astype("int64").astype(float)

    out = np.interp(mx, ax, ay)  # linear-in-log inside the anchor range (clamps outside)
    if len(yrs) >= 2:
        # replace the clamped ends with genuine linear-in-log extrapolation
        head_slope = (ay[1] - ay[0]) / (ax[1] - ax[0])
        tail_slope = (ay[-1] - ay[-2]) / (ax[-1] - ax[-2])
        head, tail = mx < ax[0], mx > ax[-1]
        out[head] = ay[0] + head_slope * (mx[head] - ax[0])
        out[tail] = ay[-1] + tail_slope * (mx[tail] - ax[-1])
    return np.exp(out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_population(
    annual: dict[str, dict[int, float]] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    out_path: str | Path | None = "population_monthly.csv",
    *,
    add_citywide: bool = True,
) -> pd.DataFrame:
    """Build the monthly population table, one row per (district, month).

    Parameters
    ----------
    annual : optional ``{district: {year: population}}`` (e.g. from
        `load_annual_from_csv`); districts missing here fall back to the seed.
    start, end : ISO ``YYYY-MM-DD`` bounds; ``end`` defaults to the current month.
    add_citywide : also emit the "Shahar bo'yicha" row as the sum of all districts.

    Returns
    -------
    DataFrame with columns: date, district, population, pop_yoy_growth,
    pop_share_city - mergeable via ``on=["district", "date"]``.
    """
    if end is None:
        end = datetime.now(timezone.utc).date().replace(day=1).isoformat()
    months = pd.date_range(start=pd.Timestamp(start).replace(day=1), end=end, freq="MS")
    years = range(pd.Timestamp(start).year, months[-1].year + 2)  # +1 anchor past the end

    seed = _seed_annual(years)
    if annual:  # merge user anchors over the seed (per district)
        for name, yearmap in annual.items():
            seed.setdefault(name, {}).update(yearmap)

    frames = []
    for name, anchors in seed.items():
        pop = _interpolate_district(anchors, months)
        frames.append(pd.DataFrame({"date": months, "district": name, "population": np.round(pop).astype("int64")}))
    pop_df = pd.concat(frames, ignore_index=True)

    if add_citywide:
        city = (
            pop_df.groupby("date", as_index=False)["population"].sum().assign(district=CITYWIDE_KEY)
        )
        pop_df = pd.concat([pop_df, city[["date", "district", "population"]]], ignore_index=True)

    # Derived features. City total per month is used for the district's share.
    city_total = pop_df.loc[pop_df["district"] == CITYWIDE_KEY].set_index("date")["population"]
    if city_total.empty:  # citywide not emitted -> derive the total from districts
        city_total = pop_df.groupby("date")["population"].sum()
    pop_df = pop_df.sort_values(["district", "date"]).reset_index(drop=True)
    pop_df["pop_yoy_growth"] = (
        pop_df.groupby("district")["population"].pct_change(12) * 100.0
    )
    pop_df["pop_share_city"] = pop_df["population"] / pop_df["date"].map(city_total)

    pop_df = pop_df[["date", "district", "population", "pop_yoy_growth", "pop_share_city"]]
    pop_df = pop_df.sort_values(["district", "date"]).reset_index(drop=True)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pop_df.to_csv(out_path, index=False)
        print(
            f"[population] wrote {len(pop_df)} rows ({pop_df['district'].nunique()} districts x "
            f"{pop_df['date'].nunique()} months) -> {out_path}"
        )
    return pop_df


def main() -> None:
    pop = build_population()
    print(pop.query("date == date.max()").to_string(index=False))


if __name__ == "__main__":
    main()


# Example usage (in FoodTashkent.ipynb):
#     from utils.population import build_population, load_annual_from_csv
#     # With official numbers (recommended):
#     #   population_annual.csv has columns: district,year,population
#     anchors = load_annual_from_csv("datasets/population_annual.csv")
#     pop = build_population(anchors)                      # -> population_monthly.csv
#     df = df.merge(pop, on=["district", "date"], how="left")
