#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a node-aware multimodal network (road + rail + connection)
with *local* per-vertex explosion only where needed (around connections).

Why this is fast:
  • We still split only the road/rail lines that a connection touches.
  • We only "edgeletize" (explode into per-vertex segments) those impacted lines,
    not the entire country.

Outputs (EPSG:25832):
  graphs/multimodal_network.gpkg
    - layer 'network_edges' (LineString): columns [u, v, mode, conn_type?]
    - layer 'network_nodes' (Point):      columns [node_id]

Usage (defaults work):
  python build_multimodal_network_local_edgelets.py \
      --roads input_files/derived/germany_all_road_edges.geojson \
      --rails input_files/derived/germany_rail_edges.geojson \
      --conns graphs/road_rail_connection_edges.geojson \
      --out   graphs/multimodal_network.gpkg \
      --explode interfaces    # options: interfaces|all|none
"""

from pathlib import Path
import sys
import warnings
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint, LinearRing, MultiLineString
from shapely.ops import split as shp_split
import numpy as np
# -------------------------- Defaults --------------------------
BASE_DIR = Path(".").resolve()
DEF_ROADS = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
DEF_RAILS = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
DEF_CONNS = BASE_DIR / "graphs"      / "road_rail_connection_edges.geojson"
DEF_OUT   = BASE_DIR / "graphs"      / "multimodal_network.gpkg"

EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"

TARGET_EPSG = 25832       # metric CRS (Germany UTM32N)
EDGE_SPLIT_TOL = 25.0     # m — max distance from conn endpoint to split target line
VERTEX_EPS    = 0.05      # m — treat as "at the endpoint" along a line
SNAP_ENDPOINT_TOL = 0.50  # m — snap split segment endpoints to requested points
TOL_JOIN = 0.10           # m — node merge tolerance (tiny; avoids false bridges)
ROUND_KEY = 6             # for mapping keys

# -------------------------- Utils --------------------------
def _ensure_crs(gdf: gpd.GeoDataFrame, fallback_epsg=4326) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn(f"Input had no CRS; assuming EPSG:{fallback_epsg}.")
        gdf = gdf.set_crs(epsg=fallback_epsg, allow_override=True)
    return gdf

def _to_target_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs and gdf.crs.to_epsg() == TARGET_EPSG:
            return gdf
    except Exception:
        pass
    return gdf.to_crs(TARGET_EPSG)

def _explode_multilines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode MultiLineStrings; normalize LinearRings -> LineStrings; keep LineStrings."""
    g = gdf.explode(index_parts=False).reset_index(drop=True)
    g = g[g.geometry.notnull()]
    def _norm(geom):
        if isinstance(geom, LinearRing):
            return LineString(geom.coords)
        return geom
    g["geometry"] = g["geometry"].apply(_norm)
    out = []
    for _, r in g.iterrows():
        geom = r.geometry
        if isinstance(geom, MultiLineString):
            for sub in geom.geoms:
                d = r.drop(labels=["geometry"]).to_dict(); d["geometry"] = sub
                out.append(d)
        else:
            out.append(r)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs).reset_index(drop=True)

def _endpts(line: LineString):
    c = list(line.coords); return (c[0][0], c[0][1]), (c[-1][0], c[-1][1])

def _nearest_edge_point(pt: Point, lines_gdf: gpd.GeoDataFrame, tol: float):
    """Nearest point on any line within tol; returns (row_index, snapped_pt, distance) or (None,None,None)."""
    bbox = pt.buffer(tol).bounds
    cand = list(lines_gdf.sindex.intersection(bbox))
    if not cand: return None, None, None
    best = (None, None, float("inf"))
    for i in cand:
        geom = lines_gdf.geometry.iloc[i]
        d_along = geom.project(pt)
        snap = geom.interpolate(d_along)
        d = snap.distance(pt)
        if d < best[2]:
            best = (i, snap, d)
    if best[0] is None or best[2] > tol:
        return None, None, None
    return best

def _collect_conn_parts(conns: gpd.GeoDataFrame):
    """Return connection parts list [(cid, attrs, LineString)] and endpoints [(cid, 'A'/'B', Point)]."""
    parts = []
    endpoints = []
    for cid, row in enumerate(conns.itertuples(index=False)):
        geom = row.geometry
        attrs = row._asdict(); attrs.pop("geometry", None)
        if isinstance(geom, LineString):
            parts.append((cid, attrs, geom))
            a, b = _endpts(geom)
            endpoints.append((cid, "A", Point(a[0], a[1])))
            endpoints.append((cid, "B", Point(b[0], b[1])))
        elif isinstance(geom, MultiLineString):
            for sub in geom.geoms:
                parts.append((cid, attrs, sub))
                a, b = _endpts(sub)
                endpoints.append((cid, "A", Point(a[0], a[1])))
                endpoints.append((cid, "B", Point(b[0], b[1])))
    return parts, endpoints

def _split_line_with_points(line: LineString, pts: list[Point], vertex_eps=VERTEX_EPS):
    """
    Split a line at requested points (projected). Always return a dict mapping
    each requested point to an exact vertex: near endpoints -> that endpoint; interior -> nearest created vertex.
    """
    if not pts: return [line], {}
    L = line.length
    if L == 0: return [line], {}
    splitters = []
    exact = {}
    start_pt = Point(list(line.coords)[0]); end_pt = Point(list(line.coords)[-1])
    for p in pts:
        d_along = line.project(p)
        if d_along <= vertex_eps:
            exact[(round(p.x, ROUND_KEY), round(p.y, ROUND_KEY))] = start_pt
        elif (L - d_along) <= vertex_eps:
            exact[(round(p.x, ROUND_KEY), round(p.y, ROUND_KEY))] = end_pt
        else:
            splitters.append(line.interpolate(d_along))
    if not splitters:
        return [line], exact
    parts = shp_split(line, MultiPoint(splitters))
    parts = [seg for seg in parts.geoms if isinstance(seg, LineString) and seg.length > 0]
    # map interior requests to nearest created vertex
    for p in pts:
        key = (round(p.x, ROUND_KEY), round(p.y, ROUND_KEY))
        if key in exact: continue
        best = (None, float("inf"))
        for seg in parts:
            for v in seg.coords:
                dv = Point(v).distance(p)
                if dv < best[1]:
                    best = (Point(v), dv)
        exact[key] = best[0]
    return parts, exact

def _force_endpoints_to_targets(parts: list[LineString], targets: list[Point], snap_tol=SNAP_ENDPOINT_TOL) -> list[LineString]:
    """If segment start/end is within snap_tol of any target, set it exactly to target coordinates."""
    if not parts or not targets: return parts
    txy = [(float(p.x), float(p.y)) for p in targets]
    out = []
    for seg in parts:
        coords = list(seg.coords)
        for idx in (0, -1):
            ex, ey = coords[idx]
            # nearest target
            tx, ty = min(txy, key=lambda xy: (ex-xy[0])**2 + (ey-xy[1])**2)
            if ((ex-tx)**2 + (ey-ty)**2) ** 0.5 <= snap_tol:
                coords[idx] = (tx, ty)
        out.append(LineString(coords))
    return out

def _explode_to_segments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Explode each LineString into per-vertex segments (edgelets).
    For coords [p0,p1,...,pn], emits (p0->p1),(p1->p2),...,(p{n-1}->pn).
    """
    rows = []
    append = rows.append
    for _, r in gdf.iterrows():
        geom: LineString = r.geometry
        if geom is None or geom.is_empty: continue
        cs = list(geom.coords)
        if len(cs) < 2: continue
        base = r.drop(labels=["geometry"]).to_dict()
        for i in range(len(cs)-1):
            a = cs[i]; b = cs[i+1]
            if a[0] == b[0] and a[1] == b[1]:
                continue
            d = base.copy()
            d["geometry"] = LineString([a, b])
            append(d)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)

def _assign_nodes(*layers: gpd.GeoDataFrame, tol=TOL_JOIN):
    """
    Node assignment with tiny tolerance merge (grid hash).
    """
    from math import floor, hypot
    cell = {}               # (ix, iy) -> [node_ids]
    nodes_xy = []           # node_id -> (x, y)
    edges_rows = []         # output edges

    def cell_key(xy):
        return (int(floor(xy[0] / tol)), int(floor(xy[1] / tol)))

    def get_node_id(xy):
        ix, iy = cell_key(xy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in cell.get((ix + dx, iy + dy), []):
                    x0, y0 = nodes_xy[nid]
                    if hypot(xy[0] - x0, xy[1] - y0) <= tol:
                        return nid
        nid = len(nodes_xy)
        nodes_xy.append((xy[0], xy[1]))
        cell.setdefault((ix, iy), []).append(nid)
        return nid

    for gdf in layers:
        for _, r in gdf.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty or geom.length == 0:
                continue
            (x0, y0), (x1, y1) = list(geom.coords)[0], list(geom.coords)[-1]
            u = get_node_id((float(x0), float(y0)))
            v = get_node_id((float(x1), float(y1)))
            d = r.drop(labels=["geometry"]).to_dict()
            d["u"] = int(u); d["v"] = int(v); d["geometry"] = geom
            edges_rows.append(d)

    edges = gpd.GeoDataFrame(edges_rows, geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")
    nodes = gpd.GeoDataFrame({
        "node_id": list(range(len(nodes_xy))),
        "geometry": [Point(xy) for xy in nodes_xy]
    }, geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")
    return edges, nodes
METRIC_EPSG = 25832  # adjust if your pipeline uses a different metric CRS

def _ensure_metric_crs(gdf, epsg=METRIC_EPSG):
    """Make sure we're in a metric CRS so .length is in meters."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    elif gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg)
    return gdf

def recompute_lengths_in_place(edges_gdf, add_km=True):
    """
    Recompute length_m from geometry for every edge (road/rail/connection).
    Use *after* you have split the lines and rebuilt connection geometries.
    """
    # Work in metric CRS so .length returns meters
    edges_m = _ensure_metric_crs(edges_gdf)
    # Geometry length per feature (handles LineString pieces correctly)
    edges_gdf["length_m"] = edges_m.geometry.length.astype(float)
    if add_km:
        edges_gdf["length_km"] = edges_gdf["length_m"] / 1000.0
    return edges_gdf

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser(description="Build multimodal network with local edgelets near connections.")
    ap.add_argument("--roads", type=str, default=str(DEF_ROADS))
    ap.add_argument("--rails", type=str, default=str(DEF_RAILS))
    ap.add_argument("--conns", type=str, default=str(DEF_CONNS))
    ap.add_argument("--out",   type=str, default=str(DEF_OUT))
    ap.add_argument("--explode", choices=["interfaces","all","none"], default="interfaces",
                    help="Explode per-vertex segments for: only impacted lines (interfaces), all, or none.")
    ap.add_argument("--edge-split-tol", type=float, default=EDGE_SPLIT_TOL)
    ap.add_argument("--snap-endpoint-tol", type=float, default=SNAP_ENDPOINT_TOL)
    ap.add_argument("--node-merge-tol", type=float, default=TOL_JOIN)
    args = ap.parse_args()

    roads_p = Path(args.roads); rails_p = Path(args.rails); conns_p = Path(args.conns)
    out_p   = Path(args.out)

    for p in (roads_p, rails_p, conns_p):
        if not p.exists():
            print(f"ERROR: missing input: {p}", file=sys.stderr); sys.exit(1)

    # 0) Load & normalize
    roads = _ensure_crs(gpd.read_file(roads_p))
    rails = _ensure_crs(gpd.read_file(rails_p))
    conns = _ensure_crs(gpd.read_file(conns_p))

    roads = _explode_multilines(_to_target_crs(roads))
    rails = _explode_multilines(_to_target_crs(rails))
    conns = _explode_multilines(_to_target_crs(conns))

    roads["__orig_idx"] = roads.index
    rails["__orig_idx"] = rails.index

    # 1) Collect connection parts & endpoints
    conn_parts, endpoints = _collect_conn_parts(conns)

    # 2) For each endpoint, find nearest road & rail line
    print(f"[1/6] Snapping {len(endpoints)} connection endpoints to nearest road/rail lines…")
    road_hits = {}  # (cid, side) -> (roads_idx, snap_pt, dist)
    rail_hits = {}  # (cid, side) -> (rails_idx, snap_pt, dist)
    _ = roads.sindex; _ = rails.sindex
    for cid, side, pt in endpoints:
        r_idx, r_pt, r_d = _nearest_edge_point(pt, roads, args.edge_split_tol)
        t_idx, t_pt, t_d = _nearest_edge_point(pt, rails, args.edge_split_tol)
        if r_idx is not None:
            road_hits[(cid, side)] = (int(r_idx), r_pt, float(r_d))
        if t_idx is not None:
            rail_hits[(cid, side)] = (int(t_idx), t_pt, float(t_d))

    # 3) Choose ROAD↔RAIL pairing per connection; collect split targets
    print("[2/6] Choosing road↔rail assignment per connection & batching split targets…")
    road_targets = defaultdict(list)  # roads_idx -> [Point,...]
    rail_targets = defaultdict(list)  # rails_idx -> [Point,...]
    conn_choice = {}  # cid -> {"A":(layer, idx, snap_pt), "B":(..)}

    def choose(cid):
        A_r = road_hits.get((cid, "A")); A_t = rail_hits.get((cid, "A"))
        B_r = road_hits.get((cid, "B")); B_t = rail_hits.get((cid, "B"))
        cand = []
        if A_r and B_t: cand.append((A_r[2] + B_t[2], ("road", A_r[0], A_r[1]), ("rail", B_t[0], B_t[1])))
        if A_t and B_r: cand.append((A_t[2] + B_r[2], ("rail", A_t[0], A_t[1]), ("road", B_r[0], B_r[1])))
        if not cand: return None
        cand.sort(key=lambda x: x[0]); return {"A": cand[0][1], "B": cand[0][2]}

    for cid, _, _ in conn_parts:
        ch = choose(cid)
        conn_choice[cid] = ch
        if ch is None: continue
        for side in ("A", "B"):
            layer, idx, spt = ch[side]
            (road_targets if layer=="road" else rail_targets)[idx].append(spt)

    # 4) Split impacted lines once per line; force endpoints to target snaps
    print(f"[3/6] Splitting impacted road lines: {len(road_targets)} / {len(roads)}")
    road_imp_idx = sorted(road_targets.keys())
    roads_imp = roads.loc[road_imp_idx]
    road_split_rows = []
    for idx, row in roads_imp.iterrows():
        pts = road_targets[idx]
        parts, _ = _split_line_with_points(row.geometry, pts, vertex_eps=VERTEX_EPS)
        parts = _force_endpoints_to_targets(parts, pts, snap_tol=args.snap_endpoint_tol)
        base = row.drop(labels=["geometry"]).to_dict()
        for seg in parts:
            d = base.copy(); d["geometry"] = seg; road_split_rows.append(d)
    roads_split = gpd.GeoDataFrame(road_split_rows, geometry="geometry", crs=roads.crs)
    roads_rest  = roads.drop(index=road_imp_idx)
    roads_new   = pd.concat([roads_rest, roads_split], ignore_index=True)
    roads_new["mode"] = "road"

    print(f"[4/6] Splitting impacted rail lines: {len(rail_targets)} / {len(rails)}")
    rail_imp_idx = sorted(rail_targets.keys())
    rails_imp = rails.loc[rail_imp_idx]
    rail_split_rows = []
    for idx, row in rails_imp.iterrows():
        pts = rail_targets[idx]
        parts, _ = _split_line_with_points(row.geometry, pts, vertex_eps=VERTEX_EPS)
        parts = _force_endpoints_to_targets(parts, pts, snap_tol=args.snap_endpoint_tol)
        base = row.drop(labels=["geometry"]).to_dict()
        for seg in parts:
            d = base.copy(); d["geometry"] = seg; rail_split_rows.append(d)
    rails_split = gpd.GeoDataFrame(rail_split_rows, geometry="geometry", crs=rails.crs)
    rails_rest  = rails.drop(index=rail_imp_idx)
    rails_new   = pd.concat([rails_rest, rails_split], ignore_index=True)
    rails_new["mode"] = "rail"

    # 5) Rebuild connection geometries: set endpoints to the exact chosen snap points
    print("[5/6] Rebuilding connection geometries on forced vertices…")
    conn_rows = []
    fixed = 0
    for cid, attrs, geom in conn_parts:
        ch = conn_choice.get(cid)
        if ch is None:
            d = attrs.copy(); d["geometry"] = geom; d["mode"] = "connection"; d["conn_type"] = "unsnapped"
            conn_rows.append(d); continue
        A_layer, A_idx, A_spt = ch["A"]; B_layer, B_idx, B_spt = ch["B"]
        coords = list(geom.coords)
        coords[0]  = (A_spt.x, A_spt.y)
        coords[-1] = (B_spt.x, B_spt.y)
        d = attrs.copy()
        d["geometry"] = LineString(coords)
        d["mode"] = "connection"; d["conn_type"] = "road_rail"
        conn_rows.append(d); fixed += 1
    conns_new = gpd.GeoDataFrame(conn_rows, geometry="geometry", crs=conns.crs)

    # 6) EXPLODE SELECTION:
    print("[6/6] Preparing edges for node assignment…")
    explode_mode = args.explode
    if explode_mode == "all":
        # explode everything (slow)
        roads_proc = _explode_to_segments(roads_new).assign(mode="road")
        rails_proc = _explode_to_segments(rails_new).assign(mode="rail")
    elif explode_mode == "interfaces":
        # explode only the impacted lines; keep the rest intact (fast)
        roads_edgelets = _explode_to_segments(roads_split).assign(mode="road")
        rails_edgelets = _explode_to_segments(rails_split).assign(mode="rail")
        # untouched remainder stays as-is
        roads_proc = pd.concat([roads_rest.assign(mode="road"), roads_edgelets], ignore_index=True)
        rails_proc = pd.concat([rails_rest.assign(mode="rail"), rails_edgelets], ignore_index=True)
    else:  # "none" — just the split geometries, no extra edgelets
        roads_proc = roads_new.assign(mode="road")
        rails_proc = rails_new.assign(mode="rail")

    conns_proc = conns_new.assign(mode="connection")

    # Assign node IDs with tiny tolerance merge
    edges_all, nodes_all = _assign_nodes(roads_proc, rails_proc, conns_proc, tol=args.node_merge_tol)

    # Diagnostics
    edges_all["u"] = edges_all["u"].astype(int)
    edges_all["v"] = edges_all["v"].astype(int)
    edges_all["mode"] = edges_all["mode"].astype(str).str.lower()
    mcounts = edges_all["mode"].value_counts().to_dict()
    acc  = edges_all[edges_all["mode"].isin(["road","connection"])]
    rail = edges_all[edges_all["mode"]=="rail"]
    acc_nodes  = set(pd.concat([acc["u"],  acc["v"]], ignore_index=True).to_numpy()) if len(acc)  else set()
    rail_nodes = set(pd.concat([rail["u"], rail["v"]], ignore_index=True).to_numpy()) if len(rail) else set()
    iface = acc_nodes & rail_nodes
    conn_only = edges_all[edges_all["mode"]=="connection"]
    hits_u = conn_only["u"].isin(iface).sum()
    hits_v = conn_only["v"].isin(iface).sum()
    hits_b = ((conn_only["u"].isin(iface)) & (conn_only["v"].isin(iface))).sum()

    # Write
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists(): out_p.unlink()
    edges_all.to_file(out_p, layer=EDGES_LAYER, driver="GPKG")
    nodes_all.to_file(out_p, layer=NODES_LAYER, driver="GPKG")

    print("✅ Wrote", out_p)
    print(f"   Edges: {len(edges_all):,}  Nodes: {len(nodes_all):,}")
    print(f"   Mode counts: {mcounts}")
    print(f"   Connections with pairing: {fixed} / {len(conns_new)}")
    print(f"   Interface nodes (access∩rail): {len(iface)}")
    print(f"   Connection ends on interface nodes: u={hits_u}, v={hits_v}, both_ends={hits_b}")
    print("   CRS: EPSG:25832 (meters). Build your graph directly from [u,v].")
    print(f"   Explode mode: {explode_mode}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr); sys.exit(2)
