#!/usr/bin/env python3
"""
Process latest raw Overpass JSON (roads & border) into:
  - GeoJSON roads:   input_files/derived/germany_all_road_edges.geojson
  - GeoJSON border:  input_files/derived/germany_border.geojson
  - EPS map:         input_files/derived/germany_roads_map.eps

Run:
  python 02_process_roads.py [--roads-raw PATH] [--border-raw PATH] [--simplify METERS]

Requirements:
  pip install geopandas shapely pyproj matplotlib
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
from shapely.geometry import LineString
from shapely.ops import unary_union

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "input_files"
RAW_DIR = DATA_DIR / "raw_overpass"
OUT_DIR = DATA_DIR / "derived"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROADS_GEOJSON = OUT_DIR / "germany_all_road_edges.geojson"
BORDER_GEOJSON = OUT_DIR / "germany_border.geojson"
MAP_EPS = OUT_DIR / "germany_roads_map.eps"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def newest_json(prefix: str) -> Path:
    cand = sorted(RAW_DIR.glob(f"{prefix}_*.json"))
    if not cand:
        raise FileNotFoundError(f"No raw {prefix}_*.json found in {RAW_DIR}")
    return cand[-1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------

def parse_road_data(raw: Dict[str, Any]) -> gpd.GeoDataFrame:
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in raw.get("elements", []) if e["type"] == "node"}
    feats = []
    for e in raw.get("elements", []):
        if e["type"] == "way" and "nodes" in e:
            coords = [nodes[n] for n in e["nodes"] if n in nodes]
            if len(coords) >= 2:
                tags = e.get("tags", {})
                feats.append({
                    "geometry": LineString(coords),
                    "highway": tags.get("highway"),
                    "ref": tags.get("ref"),
                    "name": tags.get("name"),
                })
    gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326")
    # Add geodesic length in meters (works across all of Germany reliably)
    geod = pyproj.Geod(ellps="WGS84")
    gdf["length_m"] = gdf.geometry.apply(lambda geom: geod.geometry_length(geom))
    return gdf


def parse_border_data(raw: Dict[str, Any]) -> gpd.GeoDataFrame:
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in raw.get("elements", []) if e["type"] == "node"}
    border_lines = []
    for e in raw.get("elements", []):
        if e["type"] == "way" and "nodes" in e and len(e["nodes"]) >= 2:
            coords = [nodes[n] for n in e["nodes"] if n in nodes]
            if len(coords) >= 2:
                border_lines.append(LineString(coords))
    if not border_lines:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    lines = unary_union(border_lines)
    return gpd.GeoDataFrame(geometry=[lines], crs="EPSG:4326")


# --------------------------------------------------------------------------------------
# Exports & Plotting
# --------------------------------------------------------------------------------------

def export_geojson(gdf: gpd.GeoDataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    logging.info(f"Wrote GeoJSON → {path}")


def plot_eps(gdf_border: gpd.GeoDataFrame, gdf_roads: gpd.GeoDataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)
    gdf_border.plot(ax=ax, edgecolor="#888888", linewidth=0.6, facecolor="none")
    # Subtle styling for dense networks
    gdf_roads.plot(ax=ax, linewidth=0.25, alpha=0.9)
    ax.set_axis_off()
    ax.set_title("Germany: Motorway/Trunk/Primary/Secondary Network", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logging.info(f"Wrote EPS map → {out_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Process Overpass raw roads & border → GeoJSON + EPS")
    ap.add_argument("--roads-raw", type=Path, help="Path to roads raw JSON (defaults to newest)")
    ap.add_argument("--border-raw", type=Path, help="Path to border raw JSON (defaults to newest)")
    ap.add_argument("--simplify", type=float, default=None, help="Simplify roads by tolerance in meters (in EPSG:3035)")
    args = ap.parse_args()

    roads_raw_path = args.roads_raw or newest_json("roads")
    border_raw_path = args.border_raw or newest_json("border")

    roads_raw = load_json(roads_raw_path)
    border_raw = load_json(border_raw_path)

    gdf_roads = parse_road_data(roads_raw)
    gdf_border = parse_border_data(border_raw)

    # Optional simplification for size & speed
    if args.simplify is not None and not gdf_roads.empty:
        r3035 = gdf_roads.to_crs(3035)
        r3035["geometry"] = r3035.geometry.simplify(args.simplify)
        gdf_roads = r3035.to_crs(4326)

    export_geojson(gdf_roads, ROADS_GEOJSON)
    export_geojson(gdf_border, BORDER_GEOJSON)

    plot_eps(gdf_border, gdf_roads, MAP_EPS)


if __name__ == "__main__":
    main()
