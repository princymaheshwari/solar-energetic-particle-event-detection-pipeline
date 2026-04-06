"""
Scrape the NOAA Solar Proton Events table and build a CSV file
for evaluation. Filters to events from 1995 onward.

Source: https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/
        solar-proton-events/SEP%20page%20code.html

Output CSV columns:
    start_time  — event start (UTC)
    end_time    — event maximum/end time (UTC) [as requested by user]
    peak_flux   — >10 MeV maximum flux in pfu
    region      — NOAA active region number
    location    — solar disk location
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/"
    "interplanetary-data/solar-proton-events/SEP%20page%20code.html"
)

def parse_datetime(raw: str) -> str:
    """
    Parse NOAA datetime string into ISO format.

    Input formats:
        "1995 10/20 0825"
        "1997 11/04 1300"
        "2000 07/07/1010"  (occasional extra slash)

    Output:
        "1995-10-20 08:25:00"
    """
    raw = raw.strip()
    if not raw or raw == "N/A":
        return None

    # Clean up occasional formatting issues
    # e.g., "1979 07/07/1010" → "1979 07/07 1010"
    raw = re.sub(r"(\d{2})/(\d{4})", r"\1 \2", raw)

    # Try to extract year, month, day, time
    match = re.match(
        r"(\d{4})\s+(\d{2})/(\d{2})\s+(\d{4})", raw
    )
    if not match:
        # Try without time
        match = re.match(r"(\d{4})\s+(\d{2})/(\d{2})", raw)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day} 00:00:00"
        return None

    year, month, day, hhmm = match.groups()
    hour = hhmm[:2]
    minute = hhmm[2:]
    return f"{year}-{month}-{day} {hour}:{minute}:00"


def parse_peak_flux(raw: str) -> float:
    """
    Parse peak flux string, handling commas.

    "24,000" → 24000.0
    "12" → 12.0
    """
    raw = raw.strip().replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def scrape_sep_catalog():
    print("Downloading NOAA SEP catalog page...")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the table
    table = soup.find("table")
    if table is None:
        raise ValueError("Could not find table in page")

    rows = table.find_all("tr")
    print(f"Found {len(rows)} table rows")

    events = []

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        # Extract text from cells
        start_raw = cells[0].get_text(strip=True)
        max_raw = cells[1].get_text(strip=True)
        flux_raw = cells[2].get_text(strip=True)

        # Parse
        start_time = parse_datetime(start_raw)
        end_time = parse_datetime(max_raw)
        peak_flux = parse_peak_flux(flux_raw)

        if start_time is None or end_time is None or peak_flux is None:
            continue

        # Extract region and location if available
        region = cells[3].get_text(strip=True) if len(cells) > 3 else ""
        location = cells[4].get_text(strip=True) if len(cells) > 4 else ""

        events.append({
            "start_time": start_time,
            "end_time": end_time,
            "peak_flux_pfu": peak_flux,
            "region": region,
            "location": location,
        })

    print(f"Parsed {len(events)} total events")

    # Convert to DataFrame
    df = pd.DataFrame(events)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Filter to 1995+
    df = df[df["start_time"].dt.year >= 1995].copy()
    df = df.sort_values("start_time").reset_index(drop=True)

    print(f"Events from 1995 onward: {len(df)}")
    print(f"Year range: {df['start_time'].dt.year.min()} to "
          f"{df['start_time'].dt.year.max()}")

    # Save
    output_file = "noaa_sep_catalog_1995_2025.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Print summary by decade
    print("\nEvents per year:")
    year_counts = df["start_time"].dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count}")

    return df


if __name__ == "__main__":
    df = scrape_sep_catalog()
    print(f"\nFirst 5 events:")
    print(df.head().to_string())
    print(f"\nLast 5 events:")
    print(df.tail().to_string())