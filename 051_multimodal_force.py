#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multimodal router with robust node handling:
- Tries single-stage (time-based) first.
- If no rail is used, switches to 3-stage (access road+conn → rail-only → egress road+conn).
- Normalizes node IDs to plain ints to avoid "unhashable dict" errors.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt

# ---------- Defaults ----------
BASE = Path(".").resolve()
NETWORK = BASE / "graphs" / "multimodal_network.geojson"
OUT_GEOJSON = BASE / "graphs" / "route_edges.geojson"
OUT_PNG     = BASE / "graphs" / "route_by_mode.png"

# Berlin → Frankfurt am Main
DEFAULT_ORIG = (52.5200, 13.4050)
DEFAULT_DEST = (50.1109, 8.6821)

# EPSG codes
TARGET_EPSG = 25832   # meters (Germany UTM32)
PLOT_EPSG   = 3857

# Node equivalence grid (meters via rounding in projected CRS)
NODE_ROUND_DECIMALS = 0  # 1 m grid (robust)

# speeds (km/h) and transfer penalty (minutes)
DEFAULT_SPEED_ROAD = 70.0
DEFAULT_SPEED_RAIL = 160.0
DEFAULT_SPEED_CONN = 5.0
DEFAULT_XFER_MIN   = 6.0

# consider up to K nearest rail nodes around origin/dest
K_NEAREST_RAIL = 50

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# ---------- Helpers ----------
def _as_int_node(n):
    """Coerce various shapes into a plain int node id."""
    import numpy as np
    if isinstance(n, (int, np.integer)):
        return int(n)
    if isinstance(n, (list, tuple, np.ndarray)) and len(n) == 1:
        return int(n[0])
    if isinstance(n, dict):
        if len(n) == 1:
            return int(next(iter(n.keys())))
        raise TypeError(f"Node id is a dict with multiple keys: {n}")
    return int(n)

def load_lines(path: Path, epsg: int) -> gpd.GeoDataFrame:
    if not path.exists():
        print(f"ERROR: network file not found: {path}", file=sys.stderr)
        sys.exit(1)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    gdf = gdf.to_crs(epsg).explode(index_parts=False)
    gdf = gdf[gdf.geometry.notnull() & (gdf.geometry.geom_type == "LineString")].reset_index(drop=True)
    if "mode" not in gdf.columns:
        gdf = gdf.assign(mode="unknown")
    gdf["mode"] = gdf["mode"].astype(str).str.lower()
    return gdf

def endpoints(ls: LineString):
    c = list(ls.coords); return (c[0][0], c[0][1]), (c[-1][0], c[-1][1])

def rnd_xy(x, y, dec=NODE_ROUND_DECIMALS):
    return (round(float(x), dec), round(float(y), dec))

def build_graph(edges: gpd.GeoDataFrame) -> tuple[nx.DiGraph, dict, dict]:
    """
    Directed graph; each edge stores:
      - length_m
      - mode
      - geometry (LineString in TARGET_EPSG)
    Node IDs are ints.
    """
    G = nx.DiGraph()
    pos2id, id2pos, next_id = {}, {}, 0
    for _, r in edges.iterrows():
        a, b = endpoints(r.geometry)
        A = rnd_xy(*a); B = rnd_xy(*b)
        if A not in pos2id:
            pos2id[A] = int(next_id); id2pos[int(next_id)] = A; next_id += 1
        if B not in pos2id:
            pos2id[B] = int(next_id); id2pos[int(next_id)] = B; next_id += 1
        u, v = int(pos2id[A]), int(pos2id[B])
        L = float(r.geometry.length)
        m = r["mode"]
        data = {"length_m": L, "mode": m, "geometry": r.geometry}
        # add both directions; keep the shorter if duplicates
        for (x, y) in [(u, v), (v, u)]:
            if G.has_edge(x, y):
                if L < G[x][y]["length_m"]:
                    G[x][y].update(data)
            else:
                G.add_edge(x, y, **data)
    return G, pos2id, id2pos

def wgs_to_metric(latlon, epsg=TARGET_EPSG):
    g = gpd.GeoDataFrame(geometry=[Point(lon, lat) for (lat, lon) in latlon], crs=4326).to_crs(epsg)
    return [(p.x, p.y) for p in g.geometry]

def nearest_node(id2pos: dict, xy) -> int:
    """Return an int node id in id2pos closest to xy."""
    ids = np.array(list(id2pos.keys()), dtype=int)
    pts = np.array([id2pos[i] for i in ids], dtype=float)
    if len(ids) == 0:
        raise RuntimeError("Graph has no nodes.")
    q = np.array(xy, dtype=float)
    if KDTree is not None:
        tree = KDTree(pts)
        _, k = tree.query(q)
        return int(ids[int(k)])
    d2 = np.sum((pts - q) ** 2, axis=1)
    return int(ids[int(np.argmin(d2))])

def k_nearest_nodes(id2pos: dict, xy, k=K_NEAREST_RAIL):
    ids = np.array(list(id2pos.keys()), dtype=int)
    pts = np.array([id2pos[i] for i in ids], dtype=float)
    q = np.array(xy, dtype=float)
    if KDTree is not None:
        tree = KDTree(pts); _, idxs = tree.query(q, k=min(k, len(ids)))
        idxs = np.atleast_1d(idxs)
        return [int(ids[int(i)]) for i in idxs]
    d2 = np.sum((pts - q) ** 2, axis=1)
    order = np.argsort(d2)[:k]
    return [int(ids[int(i)]) for i in order]

def compute_edge_time_minutes(data, spd_road, spd_rail, spd_conn):
    km = data["length_m"] / 1000.0
    m = data["mode"]
    spd = spd_conn if m == "connection" else (spd_rail if m == "rail" else spd_road)
    return (km / max(spd, 1e-6)) * 60.0

def apply_time_weights(G, spd_road, spd_rail, spd_conn):
    for _, _, d in G.edges(data=True):
        d["time_min"] = compute_edge_time_minutes(d, spd_road, spd_rail, spd_conn)

def path_edges_gdf(G: nx.DiGraph, path):
    rows = []
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        rows.append({"u": int(u), "v": int(v), "mode": d["mode"], "length_m": float(d["length_m"]), "geometry": d["geometry"]})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{TARGET_EPSG}")

def concat_paths(*paths):
    out = []
    for p in paths:
        if not p: continue
        if out and p and out[-1] == p[0]:
            out.extend(p[1:])
        else:
            out.extend(p)
    return out

def plot_route(route_edges: gpd.GeoDataFrame, out_png: Path):
    rp = route_edges.to_crs(PLOT_EPSG)
    styles = {
        "road":       dict(color="gray",   linewidth=2.0, alpha=0.9, zorder=2),
        "rail":       dict(color="blue",   linewidth=2.6, alpha=0.9, linestyle="--", zorder=3),
        "connection": dict(color="orange", linewidth=3.0, alpha=1.0, zorder=4),
        "unknown":    dict(color="lightgray", linewidth=1.5, alpha=0.6, zorder=1),
    }
    fig, ax = plt.subplots(figsize=(9, 10))
    for m in ["road", "rail", "connection", "unknown"]:
        g = rp[rp["mode"] == m]
        if len(g):
            g.plot(ax=ax, label=m, **styles[m])
    # start / end markers
    try:
        first, last = rp.geometry.iloc[0], rp.geometry.iloc[-1]
        x0, y0 = list(first.coords)[0]; x1, y1 = list(last.coords)[-1]
        ax.scatter([x0], [y0], s=40, color="green", zorder=10, label="origin")
        ax.scatter([x1], [y1], s=60, color="red", marker="X", zorder=11, label="destination")
    except: pass
    minx, miny, maxx, maxy = rp.total_bounds
    padx, pady = (maxx-minx)*0.05, (maxy-miny)*0.05
    ax.set_xlim(minx-padx, maxx+padx); ax.set_ylim(miny-pady, maxy+pady)
    ax.set_axis_off()
    h, l = ax.get_legend_handles_labels(); uniq={}
    for hh, ll in zip(h, l):
        if ll and ll not in uniq: uniq[ll] = hh
    if uniq: ax.legend(uniq.values(), uniq.keys(), loc="lower left", frameon=False)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Plot saved: {out_png}")

# ---------- Multimodal routing ----------
def route_single_stage(G, o, d, transfer_pen_min):
    """Time-based single-stage with transfer penalty when mode changes."""
    def weight(u, v, data, state=[None]):
        w = data.get("time_min", data.get("length_m", 0.0))
        pm = state[0]; cur = data.get("mode", "unknown")
        if pm is not None and cur != pm:
            w += transfer_pen_min
        state[0] = cur
        return w
    weight.__defaults__ = ([None],)  # reset state
    try:
        return nx.shortest_path(G, o, d, weight=weight)
    except nx.NetworkXNoPath:
        return None

def ensure_rail_three_stage(G: nx.DiGraph, o, d, id2pos, spd_road, spd_rail, spd_conn):
    """
    Force-rail: pick start/end rail nodes and do:
      A) origin -> rail (road+connection)
      B) rail -> rail (rail-only)
      C) rail -> destination (road+connection)
    """
    # normalize node ids
    o = _as_int_node(o); d = _as_int_node(d)

    access_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data["mode"] in ("road", "connection")]
    rail_edges   = [(u, v, data) for u, v, data in G.edges(data=True) if data["mode"] == "rail"]

    if not rail_edges:
        return None, None

    G_access = nx.DiGraph(); G_access.add_edges_from(access_edges)
    G_rail   = nx.DiGraph(); G_rail.add_edges_from(rail_edges)

    # weights
    for H in (G_access, G_rail):
        for _, _, dta in H.edges(data=True):
            km = dta["length_m"] / 1000.0
            spd = spd_conn if dta["mode"] == "connection" else (spd_rail if dta["mode"] == "rail" else spd_road)
            dta["time_min"] = (km / max(spd, 1e-6)) * 60.0

    rail_nodes = list(G_rail.nodes())
    if not rail_nodes:
        return None, None

    # nearest rail nodes around o/d
    o_xy = id2pos[o]; d_xy = id2pos[d]

    def _knear_rail(xy, k=K_NEAREST_RAIL):
        ids = np.array(rail_nodes, dtype=int)
        pts = np.array([id2pos[i] for i in ids], dtype=float)
        if len(ids) == 0:
            return []
        if KDTree is not None:
            tree = KDTree(pts); _, idxs = tree.query(np.array(xy, dtype=float), k=min(k, len(ids)))
            idxs = np.atleast_1d(idxs)
            return [int(ids[int(i)]) for i in idxs]
        d2 = np.sum((pts - np.array(xy, float))**2, axis=1)
        order = np.argsort(d2)[:k]
        return [int(ids[int(i)]) for i in order]

    cand_o = _knear_rail(o_xy)
    cand_d = _knear_rail(d_xy)

    # helper: nearest node IN a given subgraph to an XY (for o/d anchor)
    def nearest_in_H(H, xy):
        ids = list(H.nodes())
        if not ids:
            return None
        pts = np.array([id2pos[i] for i in ids], dtype=float)
        q = np.array(xy, dtype=float)
        if KDTree is not None:
            tree = KDTree(pts); _, k = tree.query(q)
            return int(ids[int(k)])
        d2 = np.sum((pts - q) ** 2, axis=1)
        return int(ids[int(np.argmin(d2))])

    o_in_access = nearest_in_H(G_access, o_xy)
    d_in_access = nearest_in_H(G_access, d_xy)
    if o_in_access is None or d_in_access is None:
        return None, None

    def dijkstra(H, s, t):
        try:
            return nx.shortest_path(H, s, t, weight="time_min")
        except nx.NetworkXNoPath:
            return None

    bestT = None
    best_paths = None
    for rs in cand_o:
        for rt in cand_d:
            pA = dijkstra(G_access, o_in_access, rs)
            if not pA: continue
            pB = dijkstra(G_rail,   rs, rt)
            if not pB: continue
            pC = dijkstra(G_access, rt, d_in_access)
            if not pC: continue

            def path_time(H, path):
                return sum(H[u][v]["time_min"] for u, v in zip(path[:-1], path[1:]))

            T = path_time(G_access, pA) + path_time(G_rail, pB) + path_time(G_access, pC)

            if bestT is None or T < bestT:
                bestT = T
                best_paths = (pA, pB, pC)

    if best_paths is None:
        return None, None
    return concat_paths(*best_paths), bestT

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Multimodal routing that ensures rail usage when available.")
    ap.add_argument("--network", type=str, default=str(NETWORK))
    ap.add_argument("--orig-lat", type=float, default=DEFAULT_ORIG[0])
    ap.add_argument("--orig-lon", type=float, default=DEFAULT_ORIG[1])
    ap.add_argument("--dest-lat", type=float, default=DEFAULT_DEST[0])
    ap.add_argument("--dest-lon", type=float, default=DEFAULT_DEST[1])
    ap.add_argument("--speed-road", type=float, default=DEFAULT_SPEED_ROAD)
    ap.add_argument("--speed-rail", type=float, default=DEFAULT_SPEED_RAIL)
    ap.add_argument("--speed-conn", type=float, default=DEFAULT_SPEED_CONN)
    ap.add_argument("--transfer-penalty-min", type=float, default=DEFAULT_XFER_MIN)
    ap.add_argument("--out-geojson", type=str, default=str(OUT_GEOJSON))
    ap.add_argument("--out-png", type=str, default=str(OUT_PNG))
    args = ap.parse_args()

    # Load network & build graph
    net = load_lines(Path(args.network), TARGET_EPSG)
    G, pos2id, id2pos = build_graph(net)

    # Snap endpoints (and normalize to ints)
    [(ox, oy), (dx, dy)] = wgs_to_metric([(args.orig_lat, args.orig_lon), (args.dest_lat, args.dest_lon)])
    o = _as_int_node(nearest_node(id2pos, rnd_xy(ox, oy)))
    d = _as_int_node(nearest_node(id2pos, rnd_xy(dx, dy)))

    # Time weights
    apply_time_weights(G, args.speed_road, args.speed_rail, args.speed_conn)

    # Try single-stage first
    def single_stage_with_penalty():
        def weight(u, v, data, state=[None]):
            w = data.get("time_min", data.get("length_m", 0.0))
            pm = state[0]; cur = data.get("mode", "unknown")
            if pm is not None and cur != pm:
                w += args.transfer_penalty_min
            state[0] = cur
            return w
        weight.__defaults__ = ([None],)
        try:
            return nx.shortest_path(G, o, d, weight=weight)
        except nx.NetworkXNoPath:
            return None

    path = single_stage_with_penalty()
    use_three_stage = False
    if path:
        modes = [G[u][v]["mode"] for u, v in zip(path[:-1], path[1:])]
        if not any(m == "rail" for m in modes):
            use_three_stage = True
    else:
        use_three_stage = True

    if use_three_stage:
        print("ℹ️ Single-stage path had no rail or failed. Switching to 3-stage forced rail…")
        path2, _ = ensure_rail_three_stage(G, o, d, id2pos, args.speed_road, args.speed_rail, args.speed_conn)
        if not path2:
            print("ERROR: Could not build a rail-using route with the current network.", file=sys.stderr)
            sys.exit(2)
        path = path2

    # Build per-edge result
    route_edges = path_edges_gdf(G, path)

    # Export GeoJSON in original network CRS
    orig_crs = gpd.read_file(Path(args.network)).crs or "EPSG:4326"
    out_geo = Path(args.out_geojson)
    out_geo.parent.mkdir(parents=True, exist_ok=True)
    route_edges.to_crs(orig_crs).to_file(out_geo, driver="GeoJSON")
    print(f"✅ Route edges written: {out_geo}  (features: {len(route_edges)})")

    # Summary by mode
    by_mode_km = (route_edges.groupby("mode")["length_m"].sum() / 1000.0).sort_values(ascending=False)
    print("Lengths by mode (km):")
    for m, v in by_mode_km.items():
        print(f"  {m:11s}: {v:,.2f} km")
    total_km = route_edges["length_m"].sum() / 1000.0
    print(f"Total length: {total_km:,.2f} km")

    # Plot
    plot_route(route_edges, Path(args.out_png))

if __name__ == "__main__":
    main()
