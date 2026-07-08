import io
import re
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

# goldenpages.uz serves fuel prices from two URL patterns:
#   - past years:   /benzin-cena/archiv-benzin/<year>/
#   - current year: /benzin-cena/            (no year in the path)
CURRENT_YEAR = 2026

def fuel_url(year):
    if year == CURRENT_YEAR:
        return "https://www.goldenpages.uz/en/benzin-cena/"
    return f"https://www.goldenpages.uz/en/benzin-cena/archiv-benzin/{year}/"

# Each price table is preceded by a heading like:
#   "...as of January 3, 2024: sum per 1 liter*"
# Data-quality traps seen across 2022-2026:
#   1. Cyrillic look-alike letters ("Oсtober" -> the 'с' is Cyrillic).
#   2. Uzbek month spellings ("Aprel", "Avgust").
#   3. A misspelling ("Febrary").
#   4. The heading tag differs by page: archives use <h4>, the current page uses <h3>.
# So we normalize homoglyphs and match on the month's first 3 letters.
HOMO = str.maketrans("асеорух", "aceopyx")  # cyrillic -> latin look-alikes
MONTH = {name[:3]: num for num, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
MONTH["avg"] = 8  # Uzbek "Avgust" -> August

def parse_header_date(text):
    m = re.search(r"as of\s+(\S+)\s+(\d{1,2}),\s*(\d{4})", text.translate(HOMO))
    if not m:
        return None
    return datetime.date(int(m.group(3)), MONTH[m.group(1)[:3].lower()], int(m.group(2)))

def scrape_fuel_year(year):
    resp = requests.get(fuel_url(year), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    frames = []
    for tbl in soup.find_all("table"):
        header = tbl.find_previous(["h3", "h4"])   # h4 on archives, h3 on the current page
        if not header or "as of" not in header.get_text():
            continue
        date = parse_header_date(header.get_text())
        if date is None:
            continue
        snapshot = pd.read_html(io.StringIO(str(tbl)))[0]
        snapshot["date"] = pd.Timestamp(date)
        frames.append(snapshot)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

