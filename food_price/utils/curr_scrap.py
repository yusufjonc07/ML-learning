from pathlib import Path
import requests


def download_cbu_exchange_rates(year: int, dataset_path: str | Path = "datasets") -> Path:
    """
    Download CBU USD exchange rates for an entire year as an Excel file.

    Parameters
    ----------
    year : int
        Year to download (e.g. 2024).
    dataset_path : str | Path
        Directory where the Excel file will be saved.

    Returns
    -------
    Path
        Path to the downloaded Excel file.
    """

    dataset_path = Path(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    from_date = f"01.01.{year}"
    to_date = f"31.12.{year}"

    payload = {
        "format": "XLS",
        "FROM_MONTH": from_date,
        "TO_YEAR": to_date,
        "lang": "en",
        "rates": "",
        "date": f"{from_date} - {to_date}",
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://cbu.uz/en/currency-archive/",
    }

    response = requests.post(
        "https://cbu.uz/common/arkhiv_valut/excel.php",
        data=payload,
        headers=headers,
    )
    response.raise_for_status()

    filename = f"cbu_rates_{year}.xls"
    content_disposition = response.headers.get("Content-Disposition", "")
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[1].strip('"')

    output_path = dataset_path / filename

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path