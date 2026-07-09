from pathlib import Path
import re
import pandas as pd

# Central Bank of Uzbekistan (CBU) yearly rate exports are old-format .xls
# (OLE2/BIFF) files with a slightly malformed header, so xlrd raises
# "Workbook corruption" unless we pass ignore_workbook_corruption=True.
#
# Layout inside each file:
#   row 0: title
#   row 1: header -> "Date", then repeating PAIRS of columns:
#            "Number of currency units (<code>)"  and  "<Currency name> (<code>)"
#   row 2+: one row per date the CBU published rates (irregular: ~weekly,
#           sometimes daily during volatile periods).
# <code> is the ISO-4217 numeric code (840=USD, 978=EUR, 826=GBP, 643=RUB).
# The rate is quoted per "number of units", so we divide to get a per-1-unit
# rate (matters for currencies quoted per 100/1000, e.g. JPY).
CBU_DIR = Path("datasets/cbu")

# ISO-4217 numeric code -> short name, for the currencies we keep.
CURRENCIES = {840: "USD", 978: "EUR", 826: "GBP", 643: "RUB"}


def _read_cbu_file(path, currencies):
    """Read one yearly CBU .xls into a tidy (date + one column per currency) frame."""
    raw = pd.read_excel(
        path,
        engine="xlrd",
        engine_kwargs={"ignore_workbook_corruption": True},  # tolerate CBU's header
        header=None,
    )
    header = raw.iloc[1]              # real column names live on the second row
    data = raw.iloc[2:].reset_index(drop=True)

    out = {"date": pd.to_datetime(data[0], errors="coerce")}
    # Currency columns come in (units, rate) pairs starting at column 1,
    # so the rate lives on every even column index >= 2.
    for rate_col in range(2, raw.shape[1], 2):
        match = re.search(r"\((\d+)\)", str(header[rate_col]))
        if not match:
            continue
        code = int(match.group(1))
        if code not in currencies:
            continue
        rate = pd.to_numeric(data[rate_col], errors="coerce")
        units = pd.to_numeric(data[rate_col - 1], errors="coerce")
        out[currencies[code]] = rate / units
    return pd.DataFrame(out).dropna(subset=["date"])


def load_currency_rates(currencies=CURRENCIES, cbu_dir=CBU_DIR, freq="MS"):
    """Load all CBU yearly files and return monthly average exchange rates.

    Returns one row per month (month-start dates, `freq="MS"`) with a column
    per currency in UZS. Because the CBU publishes on irregular dates, rates
    are averaged within each month so the result aligns with the monthly
    food/fuel data and can be merged on `date`.
    """
    files = sorted(Path(cbu_dir).glob("*.xls"))
    if not files:
        raise FileNotFoundError(f"No CBU .xls files found in {cbu_dir}")

    daily = pd.concat([_read_cbu_file(f, currencies) for f in files], ignore_index=True)
    daily = daily.drop_duplicates("date").sort_values("date")

    monthly = daily.set_index("date").resample(freq).mean().reset_index()
    return monthly


# Example usage (in FoodTashkent.ipynb):
#     from utils.currency import load_currency_rates
#     curr_monthly = load_currency_rates()   # 1 row/month, columns: USD, EUR, GBP, RUB
#     merged_df = pd.merge(merged_df, curr_monthly, on="date", how="left")
