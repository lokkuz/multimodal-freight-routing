#!/usr/bin/env python3
# pip install geopandas shapely pyproj pandas

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.strtree import STRtree
import pyproj


# -----------------------------
# Defaults (edit if needed)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_ROADS = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
DEFAULT_RAILS = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
DEFAULT_POIS  = BASE_DIR / "input_files" / "final_pois.gpkg"

OUT_DIR        = BASE_DIR / "graphs"
OUT_GEOJSON    = OUT_DIR / "road_rail_connection_edges.geojson"
OUT_JSON       = OUT_DIR / "road_rail_connection_edges.json"
OUT_CSV        = OUT_DIR / "road_rail_connection_edges.csv"
OUT_POIS_GPKG  = OUT_DIR / "pois_with_connections.gpkg"      # new filtered POIs output
OUT_POIS_LAYER = "pois_connected"


# -----------------------------
# Helpers
# -----------------------------
def geodesic_length_m(geom) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    return float(geod.geometry_length(geom))


def try_read_lines_any(path: Path) -> gpd.GeoDataFrame:
    """
    Read either a GeoJSON/OGR-supported file OR a custom JSON list with
    {"geometry": [[lon,lat], ...], ...} into a LineString GeoDataFrame (EPSG:4326).
    """
    path = Path(path)
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        else:
            gdf = gdf.to_crs(4326)
        if (gdf.geom_type == "MultiLineString").any():
            gdf = gdf.explode(index_parts=False)
        return gdf
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8"))
        feats = []
        for it in data:
            coords = it.get("geometry") or it.get("coordinates")
            if not coords:
                continue
            try:
                line = LineString(coords)
            except Exception:
                continue
            props = {k: v for k, v in it.items() if k not in {"geometry", "coordinates"}}
            feats.append({"geometry": line, **props})
        if not feats:
            raise ValueError(f"No line features could be parsed from {path}")
        return gpd.GeoDataFrame(feats, crs=4326)


def extract_unique_vertices(lines: gpd.GeoDataFrame, precision: int = 6) -> gpd.GeoDataFrame:
    """
    Return unique vertex Points from a lines GeoDataFrame.
    Deduplicate by rounding lon/lat to 'precision' decimals (~0.1–0.2 m at German latitudes).
    """
    pts: List[Tuple[float, float]] = []
    for geom in lines.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords_iter = [geom.coords]
        elif geom.geom_type == "MultiLineString":
            coords_iter = [ls.coords for ls in geom.geoms]
        else:
            continue
        for coords in coords_iter:
            for x, y in coords:
                pts.append((round(float(x), precision), round(float(y), precision)))
    if not pts:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    df = pd.DataFrame(pts, columns=["lon", "lat"]).drop_duplicates()
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat, crs=4326))


def nearest_point_index(points_gdf: gpd.GeoDataFrame,
                        targets_gdf: gpd.GeoDataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    For each point in 'points_gdf', find index of the nearest target in 'targets_gdf'
    and the planar distance in meters (EPSG:3035). Returns (idx_series, dist_m_series)
    aligned to points_gdf.index. Uses sjoin_nearest if available, else STRtree.
    """
    if points_gdf.empty or targets_gdf.empty:
        return pd.Series(dtype="Int64"), pd.Series(dtype="float")

    p3035 = points_gdf.to_crs(3035)
    t3035 = targets_gdf.to_crs(3035).reset_index(drop=True)  # ensure 0..N-1 labels

    try:
        joined = gpd.sjoin_nearest(p3035, t3035[["geometry"]], how="left",
                                   distance_col="dist_m")
        idx = joined["index_right"].astype("Int64").reindex(p3035.index)
        dist = joined["dist_m"].reindex(p3035.index)
        return idx, dist
    except Exception:
        geoms = list(t3035.geometry.values)
        tree = STRtree(geoms)
        idmap = {id(g): i for i, g in enumerate(geoms)}
        idx_list: List[int] = []
        dist_list: List[float] = []
        for geom in p3035.geometry.values:
            ng = tree.nearest(geom)
            j = idmap[id(ng)]
            idx_list.append(j)
            dist_list.append(float(geom.distance(ng)))
        return (pd.Series(idx_list, index=p3035.index, dtype="Int64"),
                pd.Series(dist_list, index=p3035.index))


# -----------------------------
# Build direct road↔rail edges
# -----------------------------
def build_direct_edges(pois: gpd.GeoDataFrame,
                       road_nodes: gpd.GeoDataFrame,
                       rail_nodes: gpd.GeoDataFrame,
                       max_road_m: float,
                       max_rail_m: float,
                       dedup_pairs: bool = True) -> Tuple[gpd.GeoDataFrame, List[Dict[str, Any]], set]:
    """
    For each POI, connect the nearest ROAD node directly to the nearest RAIL node (bypassing the POI).
    Only create an edge if both nearest nodes are within thresholds.
    Returns (lines_gdf, edges_list, connected_poi_ids).
    connected_poi_ids contains every POI that met the thresholds (even if its edge was deduped).
    """
    # Choose a POI id/name to carry over
    cand_cols = ["poi_id", "id", "ID", "name", "Name", "identifier", "station", "title"]
    poi_id_col = next((c for c in cand_cols if c in pois.columns), None)
    if poi_id_col is None:
        pois = pois.copy()
        pois["poi_id"] = range(1, len(pois) + 1)
        poi_id_col = "poi_id"

    idx_road, dist_road = nearest_point_index(pois, road_nodes)
    idx_rail, dist_rail = nearest_point_index(pois, rail_nodes)

    geod = pyproj.Geod(ellps="WGS84")
    line_records: List[Dict[str, Any]] = []
    edge_records: List[Dict[str, Any]] = []
    connected_poi_ids: set = set()
    seen_pairs = set()  # for deduplication

    for i, poi_row in pois.iterrows():
        jr = idx_road.get(i)
        kr = idx_rail.get(i)
        if pd.isna(jr) or pd.isna(kr):
            continue

        jr_i = int(jr)
        kr_i = int(kr)
        if not (0 <= jr_i < len(road_nodes) and 0 <= kr_i < len(rail_nodes)):
            continue

        dr = float(dist_road.get(i))
        dk = float(dist_rail.get(i))
        if dr > max_road_m or dk > max_rail_m:
            continue

        # Mark POI as connected (met thresholds), regardless of dedup below
        connected_poi_ids.add(poi_row[poi_id_col])

        r_pt = road_nodes.iloc[jr_i].geometry
        k_pt = rail_nodes.iloc[kr_i].geometry

        if dedup_pairs:
            key = tuple(sorted([(round(r_pt.x, 6), round(r_pt.y, 6)),
                                (round(k_pt.x, 6), round(k_pt.y, 6))]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

        line = LineString([(r_pt.x, r_pt.y), (k_pt.x, k_pt.y)])
        length_m = geod.geometry_length(line)

        props = {
            "poi_id": poi_row[poi_id_col],
            "length_m": float(length_m),
            "u_mode": "road",
            "u_lon": float(r_pt.x), "u_lat": float(r_pt.y),
            "v_mode": "rail",
            "v_lon": float(k_pt.x), "v_lat": float(k_pt.y),
            "via": "poi",  # provenance info
        }
        line_records.append({"geometry": line, **props})
        edge_records.append({
            "u": {"mode": "road", "lon": props["u_lon"], "lat": props["u_lat"]},
            "v": {"mode": "rail", "lon": props["v_lon"], "lat": props["v_lat"]},
            "length_m": props["length_m"],
            "mode": "connection_direct_rr",
            "poi_id": props["poi_id"],
        })

    lines_gdf = gpd.GeoDataFrame(line_records, crs=4326)
    return lines_gdf, edge_records, connected_poi_ids


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Create direct ROAD↔RAIL connection edges per POI (bypassing the POI) and output POIs with connections."
    )
    ap.add_argument("--roads", type=Path, default=DEFAULT_ROADS, help="Roads GeoJSON or custom JSON")
    ap.add_argument("--rails", type=Path, default=DEFAULT_RAILS, help="Rails GeoJSON or custom JSON")
    ap.add_argument("--pois",  type=Path, default=DEFAULT_POIS,  help="POIs file (e.g., final_pois.gpkg)")
    ap.add_argument("--pois-layer", default=0, help="Layer name or index for the POI GPKG (default 0)")
    ap.add_argument("--max-road-km", type=float, default=15.0, help="Max distance from POI to nearest road node (km)")
    ap.add_argument("--max-rail-km", type=float, default=3.0, help="Max distance from POI to nearest rail node (km)")
    ap.add_argument("--no-dedup", action="store_true", help="Disable dedup of identical road↔rail pairs")
    ap.add_argument("--out-geojson", type=Path, default=OUT_GEOJSON)
    ap.add_argument("--out-json",    type=Path, default=OUT_JSON)
    ap.add_argument("--out-csv",     type=Path, default=OUT_CSV)
    ap.add_argument("--out-pois",    type=Path, default=OUT_POIS_GPKG, help="Output POIs (only with connections). Use .gpkg or .geojson")
    ap.add_argument("--out-pois-layer", default=OUT_POIS_LAYER, help="Layer name if writing to GPKG")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load networks and POIs
    roads = try_read_lines_any(args.roads)
    rails = try_read_lines_any(args.rails)
    road_nodes = extract_unique_vertices(roads).reset_index(drop=True)
    rail_nodes = extract_unique_vertices(rails).reset_index(drop=True)

    pois = gpd.read_file(args.pois, layer=args.pois_layer)
    pois_crs = pois.crs or "EPSG:4326"
    if pois.crs is None:
        pois = pois.set_crs(4326)
    else:
        pois = pois.to_crs(4326)

    if args.debug:
        print(f"roads: {len(roads)} lines, nodes={len(road_nodes)}")
        print(f"rails: {len(rails)} lines, nodes={len(rail_nodes)}")
        print(f"pois:  {len(pois)}")

    # Build connections
    lines_gdf, edges_list, connected_ids = build_direct_edges(
        pois,
        road_nodes,
        rail_nodes,
        max_road_m=args.max_road_km * 1000.0,
        max_rail_m=args.max_rail_km * 1000.0,
        dedup_pairs=(not args.no_dedup),
    )

    # ---- Exports: edges ----
    if not lines_gdf.empty:
        lines_gdf.to_file(args.out_geojson, driver="GeoJSON")
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(edges_list, fh, indent=2)
    if edges_list:
        df = pd.DataFrame([
            {
                "u_mode": e["u"]["mode"], "u_lon": e["u"]["lon"], "u_lat": e["u"]["lat"],
                "v_mode": e["v"]["mode"], "v_lon": e["v"]["lon"], "v_lat": e["v"]["lat"],
                "length_m": e["length_m"], "mode": e["mode"], "poi_id": e.get("poi_id"),
            }
            for e in edges_list
        ])
        df.to_csv(args.out_csv, index=False)

    # ---- Export: POIs with connections ----
    if connected_ids:
        # Match by 'poi_id' if present, else by index set we constructed
        cand_cols = ["poi_id", "id", "ID", "name", "Name", "identifier", "station", "title"]
        poi_id_col = next((c for c in cand_cols if c in pois.columns), None)
        if poi_id_col is None:
            # We manufactured poi_id in builder if needed; but if not present here,
            # align by order: create a temp poi_id to allow filtering.
            pois = pois.copy()
            pois["poi_id"] = range(1, len(pois) + 1)
            poi_id_col = "poi_id"

        pois_connected = pois[pois[poi_id_col].isin(connected_ids)].copy()
        # Restore original CRS for output
        pois_connected = pois_connected.to_crs(pois_crs)

        outp = Path(args.out_pois)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.suffix.lower() == ".geojson":
            pois_connected.to_file(outp, driver="GeoJSON")
        else:
            pois_connected.to_file(outp, driver="GPKG", layer=args.out_pois_layer)

        print(f"POIs with connections: {len(pois_connected)} → {outp}")
    else:
        print("POIs with connections: 0 (no POI passed the thresholds)")

    print(f"Connections created: {len(edges_list)}")
    print(f"GeoJSON: {args.out_geojson if not lines_gdf.empty else '(none)'}")
    print(f"JSON:    {args.out_json}")
    print(f"CSV:     {args.out_csv}")


if __name__ == "__main__":
    main()
