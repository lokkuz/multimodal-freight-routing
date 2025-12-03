#!/usr/bin/env python3
"""
Create connection edges between POIs and the nearest ROAD and RAIL network nodes,
then export:
  - GeoJSON lines: graphs/poi_connection_edges.geojson
  - JSON edges (graph-style): graphs/poi_connection_edges.json
  - CSV table: graphs/poi_connection_edges.csv

It accepts either our GeoJSON exports (recommended):
  input_files/derived/germany_all_road_edges.geojson
  input_files/derived/germany_rail_edges.geojson
or the older custom JSON lists written by `save_*_edges_to_json` (with
"geometry": [[lon,lat],...]). The loader auto-detects the format.

Run:
  python 03_build_poi_connections.py \
    --roads input_files/derived/germany_all_road_edges.geojson \
    --rails input_files/derived/germany_rail_edges.geojson \
    --pois  input_files/final_pois.gpkg \
    --pois-layer 0 \
    --max-dist-km 10

Requirements:
  pip install geopandas shapely pyproj pandas
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.strtree import STRtree
import pyproj

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROADS = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
DEFAULT_RAILS = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
DEFAULT_POIS  = BASE_DIR / "input_files" / "final_pois.gpkg"
OUT_DIR = BASE_DIR / "graphs"
OUT_GEOJSON = OUT_DIR / "poi_connection_edges.geojson"
OUT_JSON    = OUT_DIR / "poi_connection_edges.json"
OUT_CSV     = OUT_DIR / "poi_connection_edges.csv"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def geodesic_length_m(geom) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    return geod.geometry_length(geom)


def try_read_lines_any(path: Path) -> gpd.GeoDataFrame:
    """Read either a GeoJSON/any OGR-supported file OR our custom JSON list
    with {"geometry": [[lon,lat],...], ...} into a LineString GeoDataFrame (EPSG:4326)."""
    path = Path(path)
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        # explode MultiLineString to LineString
        if (gdf.geom_type == "MultiLineString").any():
            gdf = gdf.explode(index_parts=False)
        return gdf
    except Exception:
        # Fallback: custom JSON list
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        features = []
        for it in data:
            coords = it.get("geometry") or it.get("coordinates")
            if not coords:
                continue
            try:
                line = LineString(coords)
            except Exception:
                continue
            props = {k: v for k, v in it.items() if k not in {"geometry", "coordinates"}}
            features.append({"geometry": line, **props})
        if not features:
            raise ValueError(f"No line features found in {path}")
        return gpd.GeoDataFrame(features, crs=4326)


def extract_unique_vertices(lines: gpd.GeoDataFrame, precision: int = 6) -> gpd.GeoDataFrame:
    """Return unique vertex Points from a lines GeoDataFrame. Deduplicate by rounding.
    precision=6 ≈ 0.1–0.2 m at German latitudes.
    """
    pts: List[Tuple[float, float]] = []
    for geom in lines.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            coords_iter = [geom.coords]
        elif isinstance(geom, MultiLineString):
            coords_iter = [ls.coords for ls in geom.geoms]
        else:
            continue
        for coords in coords_iter:
            for x, y in coords:
                pts.append((round(float(x), precision), round(float(y), precision)))
    if not pts:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    df = pd.DataFrame(pts, columns=["lon", "lat"]).drop_duplicates()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat, crs=4326))
    return gdf

def nearest_point_index(points_gdf: gpd.GeoDataFrame,
                        targets_gdf: gpd.GeoDataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Find index of the nearest target for each point and distance in meters.
    Returns (idx_series, dist_m_series) aligned to points_gdf.index.
    """
    if points_gdf.empty or targets_gdf.empty:
        return pd.Series(dtype="Int64"), pd.Series(dtype="float")

    # Project to metric CRS; reset *targets* index to guarantee 0..N-1 labels
    p3035 = points_gdf.to_crs(3035)
    t3035 = targets_gdf.to_crs(3035).reset_index(drop=True)

    # Preferred path: sjoin_nearest
    try:
        joined = gpd.sjoin_nearest(p3035, t3035[["geometry"]], how="left", distance_col="dist_m")
        idx = joined["index_right"].astype("Int64").reindex(p3035.index)
        dist = joined["dist_m"].reindex(p3035.index)
        return idx, dist
    except Exception:
        # Fallback: STRtree with positional indices (0..N-1) to match .iloc[]
        geoms = list(t3035.geometry.values)
        tree = STRtree(geoms)
        idmap = {id(g): i for i, g in enumerate(geoms)}
        idx_list, dist_list = [], []
        for geom in p3035.geometry.values:
            ng = tree.nearest(geom)
            j = idmap[id(ng)]  # 0..N-1
            idx_list.append(j)
            dist_list.append(float(geom.distance(ng)))
        return (pd.Series(idx_list, index=p3035.index, dtype="Int64"),
                pd.Series(dist_list, index=p3035.index))



def build_connection_edges(
    pois: gpd.GeoDataFrame,
    road_nodes: gpd.GeoDataFrame,
    rail_nodes: gpd.GeoDataFrame,
    max_dist_m: float = 10_000.0,
) -> Tuple[gpd.GeoDataFrame, list]:
    """Create connection LineStrings from each POI to the nearest road node and
    nearest rail node (if within max_dist_m). Returns (geojson_lines_gdf, edges_list)
    where edges_list contains graph-style dicts with u/v nodes and length_m.
    """
    # Choose a POI id/name to carry over
    cand_cols = [
        "poi_id", "id", "ID", "name", "Name", "identifier", "station", "title"
    ]
    poi_id_col = next((c for c in cand_cols if c in pois.columns), None)
    if poi_id_col is None:
        pois = pois.copy()
        pois["poi_id"] = range(1, len(pois) + 1)
        poi_id_col = "poi_id"

    # Find nearest road/rail nodes
    idx_road, dist_road = nearest_point_index(pois, road_nodes)
    idx_rail, dist_rail = nearest_point_index(pois, rail_nodes)

    geod = pyproj.Geod(ellps="WGS84")
    line_records: List[Dict[str, Any]] = []
    edge_records: List[Dict[str, Any]] = []

    def make_edge_row(poi_row, tgt_row, conn_type: str):
        p = poi_row.geometry
        t = tgt_row.geometry
        line = LineString([(p.x, p.y), (t.x, t.y)])
        length_m = geod.geometry_length(line)
        props = {
            "poi_id": poi_row[poi_id_col],
            "connection": conn_type,
            "length_m": float(length_m),
            "u_mode": "poi",
            "u_lon": float(p.x),
            "u_lat": float(p.y),
            "v_mode": "road" if conn_type == "poi_to_road" else "rail",
            "v_lon": float(t.x),
            "v_lat": float(t.y),
        }
        line_records.append({"geometry": line, **props})
        edge_records.append({
            "u": {"mode": "poi",  "lon": props["u_lon"], "lat": props["u_lat"]},
            "v": {"mode": props["v_mode"], "lon": props["v_lon"], "lat": props["v_lat"]},
            "length_m": props["length_m"],
            "mode": "connection",
            "connection": conn_type,
            "poi_id": props["poi_id"],
        })

    # Iterate POIs and add connections under threshold
    for i, poi_row in pois.iterrows():
        # Road
        j = idx_road.get(i)
        if pd.notna(j):
            d = float(dist_road.get(i))
            if d <= max_dist_m:
                make_edge_row(poi_row, road_nodes.iloc[int(j)], "poi_to_road")
        # Rail
        k = idx_rail.get(i)
        if pd.notna(k):
            d2 = float(dist_rail.get(i))
            if d2 <= max_dist_m:
                make_edge_row(poi_row, rail_nodes.iloc[int(k)], "poi_to_rail")

    lines_gdf = gpd.GeoDataFrame(line_records, crs=4326)
    return lines_gdf, edge_records


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build POI↔(road/rail) connection edges and lengths")
    ap.add_argument("--roads", type=Path, default=DEFAULT_ROADS, help="Path to roads GeoJSON or custom JSON")
    ap.add_argument("--rails", type=Path, default=DEFAULT_RAILS, help="Path to rails GeoJSON or custom JSON")
    ap.add_argument("--pois", type=Path,  default=DEFAULT_POIS,  help="Path to POIs (e.g., final_pois.gpkg)")
    ap.add_argument("--pois-layer", default=0, help="Layer name or index for the POI GPKG (default 0)")
    ap.add_argument("--max-dist-km", type=float, default=10.0, help="Max snap distance in km")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load networks (lines) → vertices (nodes)
    roads = try_read_lines_any(args.roads)
    rails = try_read_lines_any(args.rails)

    road_nodes = extract_unique_vertices(roads)
    rail_nodes = extract_unique_vertices(rails)

    # Load POIs
    pois = gpd.read_file(args.pois, layer=args.pois_layer)
    if pois.crs is None:
        pois = pois.set_crs(4326)
    else:
        pois = pois.to_crs(4326)

    # Build connections
    lines_gdf, edges_list = build_connection_edges(
        pois, road_nodes, rail_nodes, max_dist_m=args.max_dist_km * 1000.0
    )

    # Exports
    if not lines_gdf.empty:
        lines_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(edges_list, fh, indent=2)
    # CSV (flat schema)
    if edges_list:
        df = pd.DataFrame([
            {
                "u_mode": e["u"]["mode"],
                "u_lon": e["u"]["lon"],
                "u_lat": e["u"]["lat"],
                "v_mode": e["v"]["mode"],
                "v_lon": e["v"]["lon"],
                "v_lat": e["v"]["lat"],
                "mode": e["mode"],
                "connection": e.get("connection"),
                "length_m": e["length_m"],
                "poi_id": e.get("poi_id"),
            }
            for e in edges_list
        ])
        df.to_csv(OUT_CSV, index=False)

    print(f"Connections: {len(edges_list)}")
    print(f"GeoJSON: {OUT_GEOJSON if not lines_gdf.empty else '(none)'}")
    print(f"JSON:    {OUT_JSON}")
    print(f"CSV:     {OUT_CSV}")


if __name__ == "__main__":
    main()
