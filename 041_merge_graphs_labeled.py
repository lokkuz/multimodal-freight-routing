#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import warnings
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from shapely.ops import split

# --------------------------
# CONFIG
# --------------------------

BASE_DIR = Path(".").resolve()

ROADS_F = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_F = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS_F = BASE_DIR / "graphs"      / "road_rail_connection_edges.geojson"

OUT_GPKG = BASE_DIR / "graphs" / "multimodal_network.gpkg"  # keeps high precision
EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"

# Work in metric CRS to avoid precision issues (ETRS89 / UTM32N)
TARGET_EPSG = 25832

# Snapping radii
TOLS_PRIMARY  = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # initial search
TOLS_FALLBACK = [40.0, 50.0, 60.0]                   # if one side missing

# When splitting, ignore points closer than this to an existing vertex (meters)
VERTEX_EPS = 0.10

# --------------------------
# HELPERS
# --------------------------

def _ensure_crs(gdf: gpd.GeoDataFrame, fallback_epsg=4326) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn(f"Input had no CRS; assuming EPSG:{fallback_epsg}.")
        gdf = gdf.set_crs(epsg=fallback_epsg, allow_override=True)
    return gdf

def _to_target_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    try:
        if gdf.crs and gdf.crs.to_epsg() == epsg:
            return gdf
    except Exception:
        pass
    return gdf.to_crs(epsg=epsg)

def _explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf.explode(index_parts=False).reset_index(drop=True)
    g = g[g.geometry.notnull()]
    g = g[g.geometry.geom_type.isin(["LineString", "LinearRing", "MultiLineString"])]
    def _ring_to_line(geom):
        if geom.geom_type == "LinearRing":
            return LineString(geom.coords)
        return geom
    g["geometry"] = g["geometry"].apply(_ring_to_line)
    return g

def _endpoints_of_linestring(line: LineString):
    c = list(line.coords); return Point(c[0]), Point(c[-1])

def _nearest_line_hit(point: Point, network_gdf: gpd.GeoDataFrame, tol: float):
    _ = network_gdf.sindex
    bbox = point.buffer(tol).bounds
    cand_idx = list(network_gdf.sindex.intersection(bbox))
    best = (None, None, float("inf"))
    for idx in cand_idx:
        line = network_gdf.geometry.iloc[idx]
        proj = line.project(point)
        snapped = line.interpolate(proj)
        d = snapped.distance(point)
        if d < best[2]:
            best = (idx, snapped, d)
    if best[0] is None or best[2] > tol:
        return None, None, None
    return best

def _best_hit(pt, roads, rails):
    """Find nearest road AND nearest rail around pt using adaptive radii; return both (may be None)."""
    a_road = (None, None, None)
    a_rail = (None, None, None)
    for tol in TOLS_PRIMARY:
        if a_road[0] is None:
            a_road = _nearest_line_hit(pt, roads, tol)
            if a_road[0] is None: a_road = (None, None, None)
        if a_rail[0] is None:
            a_rail = _nearest_line_hit(pt, rails, tol)
            if a_rail[0] is None: a_rail = (None, None, None)
        if a_road[0] is not None and a_rail[0] is not None:
            break
    return a_road, a_rail

def _split_line_with_points(line: LineString, pts: list[Point], vertex_eps=0.05) -> list[LineString]:
    if not pts or line.length == 0: return [line]
    splitters = []
    coords = list(line.coords)
    for p in pts:
        d = line.project(p)
        if d <= vertex_eps or (line.length - d) <= vertex_eps:
            continue
        if min(Point(xy).distance(p) for xy in coords) <= vertex_eps:
            continue
        splitters.append(line.interpolate(d))
    if not splitters:
        return [line]
    try:
        parts = split(line, MultiPoint(splitters))
        return [seg for seg in parts.geoms if isinstance(seg, LineString) and seg.length > 0]
    except Exception:
        return [line]

def _group_split_points_by_line(points_on_network, network_gdf):
    by_line = {}
    for li, pt in points_on_network:
        by_line.setdefault(li, []).append(pt)
    # deduplicate per line by ~1e-6 deg/metric
    for li, pts in by_line.items():
        uniq, seen = [], set()
        for p in pts:
            key = (round(p.x, 6), round(p.y, 6))
            if key not in seen:
                seen.add(key); uniq.append(p)
        by_line[li] = uniq
    return by_line

def _attach_and_split(network_gdf: gpd.GeoDataFrame, conn_pts_snaps: list, vertex_eps: float) -> gpd.GeoDataFrame:
    by_line = _group_split_points_by_line(conn_pts_snaps, network_gdf)
    out_rows = []
    # We also need to know the EXACT vertex coordinates after splitting
    split_vertex_registry = {}  # (line_idx, approx_point_key) -> exact_vertex_point
    for idx, row in network_gdf.iterrows():
        geom = row.geometry
        pts = by_line.get(idx, [])
        if not pts or geom is None:
            d = row.to_dict(); d["geometry"] = geom
            out_rows.append(d); continue
        # Split
        if isinstance(geom, LineString):
            parts = _split_line_with_points(geom, pts, vertex_eps)
            # collect exact vertices created that coincide with snap points
            for p in pts:
                # find closest vertex among parts to p
                best = (None, float("inf"))
                for part in parts:
                    for v in list(part.coords):
                        dv = Point(v).distance(p)
                        if dv < best[1]:
                            best = (Point(v), dv)
                if best[0] is not None:
                    split_vertex_registry[(idx, (round(p.x,6), round(p.y,6)))] = best[0]
            for part in parts:
                out_rows.append(_row_to_dict(row, part))
        elif isinstance(geom, MultiLineString):
            new_parts = []
            for sub in geom.geoms:
                sub_parts = _split_line_with_points(sub, pts, vertex_eps)
                new_parts.extend(sub_parts)
                for p in pts:
                    best = (None, float("inf"))
                    for part in sub_parts:
                        for v in list(part.coords):
                            dv = Point(v).distance(p)
                            if dv < best[1]:
                                best = (Point(v), dv)
                    if best[0] is not None:
                        split_vertex_registry[(idx, (round(p.x,6), round(p.y,6)))] = best[0]
            out_rows.append(_row_to_dict(row, MultiLineString(new_parts)))
        else:
            d = row.to_dict(); d["geometry"] = geom
            out_rows.append(d)

    gdf = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=network_gdf.crs).reset_index(drop=True)
    return gdf, split_vertex_registry

def _row_to_dict(row, new_geom):
    d = row.drop(labels=["geometry"]).to_dict()
    d["geometry"] = new_geom
    return d

# --------------------------
# MAIN
# --------------------------

def main():
    # Load
    if not ROADS_F.exists(): print(f"ERROR: Roads file not found: {ROADS_F}", file=sys.stderr); sys.exit(1)
    if not RAILS_F.exists(): print(f"ERROR: Rails file not found: {RAILS_F}", file=sys.stderr); sys.exit(1)
    if not CONNS_F.exists(): print(f"ERROR: Connections file not found: {CONNS_F}", file=sys.stderr); sys.exit(1)

    roads = _ensure_crs(gpd.read_file(ROADS_F))
    rails = _ensure_crs(gpd.read_file(RAILS_F))
    conns = _ensure_crs(gpd.read_file(CONNS_F))

    # Work in metric CRS (stay in EPSG:25832 to preserve exact vertices)
    roads = _to_target_crs(roads, TARGET_EPSG)
    rails = _to_target_crs(rails, TARGET_EPSG)
    conns = _to_target_crs(conns, TARGET_EPSG)

    roads = _explode_lines(roads)
    rails = _explode_lines(rails)
    conns = _explode_lines(conns)

    # For each connection PART, find a road↔rail pairing
    _ = roads.sindex; _ = rails.sindex

    conn_pairs = []   # list of dicts per connection part: {a_side:('road'/'rail', idx, snapped_pt), b_side:..., ok:bool}
    conn_parts = []   # (row, part_geom, a_pt, b_pt)
    for _, row in conns.iterrows():
        geom = row.geometry
        parts = [geom] if isinstance(geom, LineString) else list(geom.geoms)
        for part in parts:
            a_pt, b_pt = _endpoints_of_linestring(part)
            conn_parts.append((row, part, a_pt, b_pt))
            # nearest for each endpoint to both layers
            a_road, a_rail = _best_hit(a_pt, roads, rails)
            b_road, b_rail = _best_hit(b_pt, roads, rails)

            # candidates: (A->road,B->rail) vs (A->rail,B->road), pick min total dist
            cand = []
            if a_road[0] is not None and b_rail[0] is not None:
                cand.append( (a_road[2] + b_rail[2], ("road", a_road[0], a_road[1]), ("rail", b_rail[0], b_rail[1])) )
            if a_rail[0] is not None and b_road[0] is not None:
                cand.append( (a_rail[2] + b_road[2], ("rail", a_rail[0], a_rail[1]), ("road", b_road[0], b_road[1])) )

            if not cand:
                # try fallback expansions on missing side(s)
                if a_rail[0] is None:
                    for tol in TOLS_FALLBACK:
                        idx, spt, dist = _nearest_line_hit(a_pt, rails, tol)
                        if idx is not None: a_rail = (idx, spt, dist); break
                if a_road[0] is None:
                    for tol in TOLS_FALLBACK:
                        idx, spt, dist = _nearest_line_hit(a_pt, roads, tol)
                        if idx is not None: a_road = (idx, spt, dist); break
                if b_rail[0] is None:
                    for tol in TOLS_FALLBACK:
                        idx, spt, dist = _nearest_line_hit(b_pt, rails, tol)
                        if idx is not None: b_rail = (idx, spt, dist); break
                if b_road[0] is None:
                    for tol in TOLS_FALLBACK:
                        idx, spt, dist = _nearest_line_hit(b_pt, roads, tol)
                        if idx is not None: b_road = (idx, spt, dist); break
                if a_road[0] is not None and b_rail[0] is not None:
                    cand.append( (a_road[2] + b_rail[2], ("road", a_road[0], a_road[1]), ("rail", b_rail[0], b_rail[1])) )
                if a_rail[0] is not None and b_road[0] is not None:
                    cand.append( (a_rail[2] + b_road[2], ("rail", a_rail[0], a_rail[1]), ("road", b_road[0], b_road[1])) )

            if cand:
                cand.sort(key=lambda x: x[0])
                _, A, B = cand[0]
                conn_pairs.append({"A": A, "B": B, "ok": True})
            else:
                conn_pairs.append({"A": (None,None,None), "B": (None,None,None), "ok": False})

    # Split roads/rails only at used snap points. While splitting, capture the EXACT vertex coordinate created.
    road_hits = []
    rail_hits = []
    for pair in conn_pairs:
        if not pair["ok"]: continue
        A, B = pair["A"], pair["B"]
        for side in (A, B):
            layer, idx, spt = side
            if layer == "road":
                road_hits.append((idx, spt))
            elif layer == "rail":
                rail_hits.append((idx, spt))

    roads_split, road_vertex_exact = _attach_and_split(roads, road_hits, VERTEX_EPS)
    rails_split, rail_vertex_exact = _attach_and_split(rails, rail_hits, VERTEX_EPS)

    # Now rebuild connection lines so their endpoints are set to the EXACT split vertex coordinates,
    # and remember these exact coordinates to build identical node IDs.
    new_conn_rows = []
    good_pairs = 0
    for (base_row, part_geom, a_pt, b_pt), pair in zip(conn_parts, conn_pairs):
        if not pair["ok"]:
            d = base_row.to_dict(); d["geometry"] = part_geom; d["mode"] = "connection"; d["conn_type"] = "unsnapped"
            new_conn_rows.append(d)
            continue

        # Resolve A and B exact vertices
        def exact_vertex(side, orig_pt):
            layer, idx, snap = side
            key = (idx, (round(snap.x,6), round(snap.y,6)))
            if layer == "road":
                return road_vertex_exact.get(key, snap)
            else:
                return rail_vertex_exact.get(key, snap)

        A_exact = exact_vertex(pair["A"], a_pt)
        B_exact = exact_vertex(pair["B"], b_pt)

        # Replace endpoints with exact vertices
        coords = list(part_geom.coords)
        coords[0]  = (A_exact.x, A_exact.y)
        coords[-1] = (B_exact.x, B_exact.y)
        new_geom = LineString(coords)

        d = base_row.to_dict()
        d["geometry"] = new_geom
        d["mode"] = "connection"
        d["conn_type"] = "road_rail"
        new_conn_rows.append(d)
        good_pairs += 1

    conns_snapped = gpd.GeoDataFrame(new_conn_rows, geometry="geometry", crs=conns.crs).reset_index(drop=True)

    # Label modes for roads/rails
    roads_split = roads_split.copy(); roads_split["mode"] = "road"
    rails_split = rails_split.copy(); rails_split["mode"] = "rail"

    # --------------------------
    # Build NODE REGISTRY with exact coordinates
    # --------------------------
    def endpoints_xy(ls: LineString): c=list(ls.coords); return (c[0][0], c[0][1]), (c[-1][0], c[-1][1])

    node_to_id = {}   # (x,y) -> id
    id_to_pt  = []    # index -> (x,y)
    def get_node_id(xy):
        nid = node_to_id.get(xy)
        if nid is not None:
            return nid
        nid = len(id_to_pt)
        node_to_id[xy] = nid
        id_to_pt.append(xy)
        return nid

    # First pass: ensure that all endpoints from roads_split + rails_split define the canonical nodes
    for gdf in (roads_split, rails_split):
        for geom in gdf.geometry:
            if geom is None: continue
            a, b = endpoints_xy(geom)
            get_node_id(a); get_node_id(b)

    # Second pass: assign u/v to all edges using the EXACT endpoints
    def as_edges(gdf, extra_cols: dict):
        rows = []
        for _, r in gdf.iterrows():
            geom = r.geometry
            if geom is None or geom.length == 0: continue
            a, b = endpoints_xy(geom)
            u = get_node_id(a); v = get_node_id(b)
            d = r.drop(labels=["geometry"]).to_dict()
            d.update(extra_cols)
            d["u"] = int(u); d["v"] = int(v)
            d["geometry"] = geom
            rows.append(d)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)

    edges_road = as_edges(roads_split, {"mode": "road"})
    edges_rail = as_edges(rails_split, {"mode": "rail"})
    edges_conn = as_edges(conns_snapped, {"mode": "connection"})

    edges = pd.concat([edges_road, edges_rail, edges_conn], ignore_index=True)

    # Nodes GeoDataFrame
    nodes = gpd.GeoDataFrame({
        "node_id": list(range(len(id_to_pt))),
        "geometry": [Point(xy) for xy in id_to_pt]
    }, geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")

    # --------------------------
    # Write outputs (keep EPSG:25832 for precision)
    # --------------------------
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    # overwrite layers if file exists
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    edges.to_file(OUT_GPKG, layer=EDGES_LAYER, driver="GPKG")
    nodes.to_file(OUT_GPKG, layer=NODES_LAYER, driver="GPKG")

    # Stats
    total_conns = len(conns_snapped)
    rr = (edges_conn.get("conn_type", pd.Series(["road_rail"]*len(edges_conn))).value_counts()
          if "conn_type" in edges_conn.columns else pd.Series())
    print(f"✅ Wrote {OUT_GPKG}")
    print(f"   Edges: {len(edges)} (road={len(edges_road)}, rail={len(edges_rail)}, connection={len(edges_conn)})")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Connection parts: {total_conns} | road↔rail formed: {good_pairs} | unsnapped kept: {total_conns - good_pairs}")
    if not rr.empty:
        print("   connection types:", rr.to_dict())
    print("   CRS: EPSG:25832 (meters)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)
