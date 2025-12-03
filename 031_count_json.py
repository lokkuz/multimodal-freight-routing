#!/usr/bin/env python3
# 03_count_json.py
# pip install geopandas  # (not required here, only json/stdlib used)

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

# --- defaults (edit to taste) ---
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PATHS = [
    BASE_DIR / "input_files" / "input_files" / "input_files" / "germany_all_road_edges.json",  # your road file
    BASE_DIR / "input_files" / "derived" / "germany_rail_edges.json",                          # typical rail path
    BASE_DIR / "graphs"     / "road_rail_connection_edges.json",                               # typical connections path
]

# Keys we might summarize if --by-type is set (first one present will be used)
TYPE_KEYS = ["highway", "railway", "mode", "type"]

def load_edges(p: Path) -> List[Dict[str, Any]]:
    """Load your custom JSON edges file. Top-level list of dicts is expected,
    but we also handle common alternatives gracefully."""
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Missing file: {p}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to read {p}: {e}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # common wrappers
        for k in ("edges", "items", "data", "lines", "features"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        # dict of id -> edge
        return [v for v in data.values() if isinstance(v, dict) and "geometry" in v]
    return []

def is_valid_edge(e: Dict[str, Any]) -> bool:
    """Edge is valid if it has geometry with at least 2 coordinates."""
    geom = e.get("geometry")
    if not isinstance(geom, list):
        return False
    return len(geom) >= 2

def summarize_types(edges: List[Dict[str, Any]]):
    """Print counts by first matching key among TYPE_KEYS."""
    key = next((k for k in TYPE_KEYS if any(k in e for e in edges)), None)
    if not key:
        print("  (no recognizable type key among: " + ", ".join(TYPE_KEYS) + ")")
        return
    c = Counter((e.get(key) or "(none)") for e in edges)
    for k, v in c.most_common():
        print(f"  {key}={k}: {v:,}")

def main():
    ap = argparse.ArgumentParser(description="Count edges in custom JSON edge files.")
    ap.add_argument("paths", nargs="*", help="Path(s) to .json files. If omitted, defaults are used.")
    ap.add_argument("--valid-only", action="store_true",
                    help="Only count entries that have geometry with ≥ 2 points.")
    ap.add_argument("--by-type", action="store_true",
                    help="Also print counts grouped by 'highway'/'railway'/'mode'.")
    args = ap.parse_args()

    # Use provided paths or fall back to defaults
    path_strs = args.paths if args.paths else [str(p) for p in DEFAULT_PATHS]
    if not args.paths:
        print("[INFO] No paths given; using defaults:")
        for p in DEFAULT_PATHS:
            print("       -", p)

    grand_total = 0
    for s in path_strs:
        p = Path(s)
        edges = load_edges(p)
        if args.valid_only:   # <-- underscore
            edges = [e for e in edges if is_valid_edge(e)]

        print(f"{p.name}: {len(edges):,} edges")
        grand_total += len(edges)

        if args.by_type and edges:  # <-- underscore
            summarize_types(edges)

    if len(path_strs) > 1:
        print("-" * 32)
        print(f"Total across files: {grand_total:,} edges")


if __name__ == "__main__":
    main()
