#!/usr/bin/env python3
"""
Fetch Germany rails and border from Overpass and save raw JSON snapshots.

Usage:
  python overpass_fetch.py fetch [--include-service]

Outputs (timestamped):
  input_files/raw_overpass/rails_YYYYmmddTHHMMSSZ.json
  input_files/raw_overpass/border_YYYYmmddTHHMMSSZ.json

Requirements:
  pip install requests

Tip: Run this first, then run process_geodata.py to parse/export GeoJSON & EPS.
"""
import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, Any

import requests

# --------------------------------------------------------------------------------------
# Paths & Config
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "input_files"
RAW_DIR = DATA_DIR / "raw_overpass"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Overpass endpoints (fallback list)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

REQUEST_TIMEOUT = 180  # seconds per request
MAX_RETRIES_PER_ENDPOINT = 2
BACKOFF_SECONDS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------------------
def overpass_query_rails(kinds=("rail",), include_service=False) -> str:
    kinds_re = "|".join(kinds)
    service_filter = "" if include_service else '["service"!~"."]'
    return f"""
    [out:json][timeout:1800];
    area["ISO3166-1"="DE"]->.searchArea;
    (
      way["railway"~"^({kinds_re})$"]{service_filter}["railway:traffic_mode"!="passenger"](area.searchArea);
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
# Networking (fetch stage)
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


def main():
    p = argparse.ArgumentParser(description="Fetch raw Overpass JSON (Germany rails & border)")
    p.add_argument("--include-service", action="store_true",
                   help="Include service tracks (yard/siding/spur, etc.)")
    args = p.parse_args()

    rails_q = overpass_query_rails(include_service=args.include_service)
    border_q = overpass_query_border()
    fetch_and_save_raw("rails", rails_q)
    fetch_and_save_raw("border", border_q)



if __name__ == "__main__":
    main()
