#!/usr/bin/env python3
"""
Fetch German road network (motorway, trunk, primary, secondary + their *_link) and border
from Overpass API and save timestamped raw JSON snapshots.

Run:
  python 01_overpass_fetch_roads.py [--extra-highways HW1,HW2,...]

Examples:
  python 01_overpass_fetch_roads.py
  python 01_overpass_fetch_roads.py --extra-highways tertiary,tertiary_link

Outputs:
  input_files/raw_overpass/roads_YYYYmmddTHHMMSSZ.json
  input_files/raw_overpass/border_YYYYmmddTHHMMSSZ.json

Requirements:
  pip install requests
"""
import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import requests

# --------------------------------------------------------------------------------------
# Paths & Config
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "input_files"
RAW_DIR = DATA_DIR / "raw_overpass"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
REQUEST_TIMEOUT = 3600
MAX_RETRIES_PER_ENDPOINT = 2
BACKOFF_SECONDS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------------------
DEFAULT_HIGHWAYS = [
    "motorway","motorway_link",
    "trunk","trunk_link",
    "primary","primary_link",
    "secondary","secondary_link",
]

def overpass_query_roads(highways: List[str]) -> str:
    clauses = "\n".join([f'  way["highway"="{h}"](area.searchArea);' for h in highways])
    return f"""
    [out:json][timeout:3600];
    area["ISO3166-1"="DE"]->.searchArea;
    (
{clauses}
    );
    out body;
    >;
    out skel qt;
    """

def overpass_query_border() -> str:
    return """
    [out:json][timeout:1800];
    rel["name"="Deutschland"]["admin_level"="2"];
    out body;
    >;
    out skel qt;
    """

# --------------------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------------------

def post_overpass(query: str) -> Dict[str, Any]:
    last_err = None
    for url in OVERPASS_ENDPOINTS:
        for attempt in range(1, MAX_RETRIES_PER_ENDPOINT + 1):
            try:
                logging.info(f"POST {url} (attempt {attempt})")
                r = requests.post(url, data={"data": query}, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                logging.warning(f"Failed: {e}. Backing off {BACKOFF_SECONDS}s...")
                import time as _t; _t.sleep(BACKOFF_SECONDS)
    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_err}")


def fetch_and_save_raw(name: str, query: str) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_DIR / f"{name}_{ts}.json"
    data = post_overpass(query)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    logging.info(f"Saved raw Overpass JSON → {out_path}")
    return out_path

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Fetch raw Overpass JSON for roads & border (Germany)")
    p.add_argument("--extra-highways", type=str, default="", help="Comma-separated extra highway values to include (e.g., tertiary,tertiary_link)")
    args = p.parse_args()

    highways = DEFAULT_HIGHWAYS.copy()
    if args.extra_highways:
        extras = [s.strip() for s in args.extra_highways.split(",") if s.strip()]
        highways.extend(extras)

    roads_q = overpass_query_roads(highways)
    border_q = overpass_query_border()

    fetch_and_save_raw("roads", roads_q)
    fetch_and_save_raw("border", border_q)


if __name__ == "__main__":
    main()
