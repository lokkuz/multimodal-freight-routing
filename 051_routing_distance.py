#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distance-only routing on a multimodal network with explicit node IDs.

Inputs (GPKG):
  - layer 'network_edges': u, v, mode in {'road','rail','connection'}, geometry (LineString, EPSG:25832)
  - layer 'network_nodes': node_id, geometry (Point, EPSG:25832)

Features:
  * Single-stage shortest path by total length (meters).
  * Optional 3-stage forced-rail (access road+conn → rail-only → access road+conn), also by distance.
  * Exports per-edge route GeoJSON (WGS84) + PNG colored by mode.

Chooser (like compare script):
  * --orig / --dest city names (small built-in gazetteer; WGS84 lon/lat)
  * --orig-xy / --dest-xy with --crs 4326 or 25832
  * --orig-name / --dest-name override labels used in filenames

Default output (if --out-geojson not given):
  graphs/routes/multimodal_<orig>_to_<dest>.geojson
"""

from pathlib import Path
import sys
import argparse
import unicodedata

import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import matplotlib.pyplot as plt

# ---------------- Paths & layers ----------------
BASE = Path(".").resolve()
NETWORK_GPKG = BASE / "graphs" / "multimodal_network.gpkg"
EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"
DEFAULT_OUT_DIR = BASE / "graphs" / "routes"

# ---------------- CRS ----------------
NET_EPSG  = 25832    # metric routing CRS (UTM32N)
PLOT_EPSG = 3857     # web mercator for quick PNG preview

# ---------------- Defaults (chooser) ----------------
CITY_WGS84 = {  # lon, lat (WGS84)
    "berlin":     (13.4050, 52.5200),
    "hamburg":    (9.9937, 53.5511),
    "munich":     (11.5820, 48.1351),
    "münchen":    (11.5820, 48.1351),
    "frankfurt":  (8.6821, 50.1109),
    "köln":       (6.9603, 50.9375),
    "cologne":    (6.9603, 50.9375),
    "stuttgart":  (9.1829, 48.7758),
    "düsseldorf": (6.7820, 51.2277),
    "leipzig":    (12.3731, 51.3397),
    "dresden":    (13.7373, 51.0504),
    "bremen":     (8.8017, 53.0793),
    "hannover":   (9.7320, 52.3759),
    "nuremberg":  (11.0796, 49.4521),
    "nürnberg":   (11.0796, 49.4521)
}

# Snap radius to nearest node (meters)
SNAP_RADIUS_M = 2000.0

# Forced-rail search
INTERFACE_K_NEAREST = 50

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# ---------------- Small utils ----------------
def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = "".join(c if c.isalnum() else "-" for c in s)
    s = "-".join(seg for seg in s.split("-") if seg)
    return s.lower()

def to_net_xy(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 lon/lat → NET_EPSG XY."""
    p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(NET_EPSG).iloc[0]
    return float(p.x), float(p.y)

# ---------------- IO ----------------
def load_network(gpkg_path: Path):
    if not gpkg_path.exists():
        print(f"ERROR: network not found: {gpkg_path}", file=sys.stderr)
        sys.exit(1)
    edges = gpd.read_file(gpkg_path, layer=EDGES_LAYER)
    nodes = gpd.read_file(gpkg_path, layer=NODES_LAYER)
    if edges.crs is None: edges = edges.set_crs(epsg=NET_EPSG, allow_override=True)
    if nodes.crs is None: nodes = nodes.set_crs(epsg=NET_EPSG, allow_override=True)
    if edges.crs.to_epsg() != NET_EPSG: edges = edges.to_crs(NET_EPSG)
    if nodes.crs.to_epsg() != NET_EPSG: nodes = nodes.to_crs(NET_EPSG)
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)
    edges["mode"] = edges["mode"].astype(str).str.lower()
    nodes["node_id"] = nodes["node_id"].astype(int)
    return edges, nodes

def build_graph_from_edges(edges: gpd.GeoDataFrame) -> nx.DiGraph:
    """
    Build directed graph from explicit u/v. If duplicate (u,v), keep the shorter edge.
    """
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        u = int(r["u"]); v = int(r["v"])
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        mode = r["mode"]
        length_m = float(geom.length)
        data = {"mode": mode, "geometry": geom, "length_m": length_m}
        # forward
        cur = G.get_edge_data(u, v)
        if (cur is None) or (length_m < cur.get("length_m", float("inf"))):
            G.add_edge(u, v, **data)
        # reverse
        curb = G.get_edge_data(v, u)
        if (curb is None) or (length_m < curb.get("length_m", float("inf"))):
            G.add_edge(v, u, **data)
    return G

def id2pos_from_nodes(nodes_gdf: gpd.GeoDataFrame) -> dict[int, tuple[float, float]]:
    return {int(r.node_id): (float(r.geometry.x), float(r.geometry.y)) for r in nodes_gdf.itertuples(index=False)}

def nearest_node_id(nodes_gdf: gpd.GeoDataFrame, xy, max_radius=SNAP_RADIUS_M) -> tuple[int, float]:
    ids = nodes_gdf["node_id"].to_numpy(dtype=int)
    pts = np.vstack([nodes_gdf.geometry.x.to_numpy(), nodes_gdf.geometry.y.to_numpy()]).T
    q = np.array(xy, dtype=float)
    if KDTree is not None and len(pts):
        tree = KDTree(pts); dist, idx = tree.query(q); node_id = int(ids[int(idx)])
    else:
        d2 = np.sum((pts - q)**2, axis=1); i = int(np.argmin(d2)); dist = float(np.sqrt(d2[i])); node_id = int(ids[i])
    if dist > max_radius:
        raise RuntimeError(f"Nearest node is {dist:.1f} m away (> {max_radius} m). Increase --snap-radius-m or check inputs.")
    return node_id, float(dist)

# ---------------- Converters ----------------
def path_to_gdf(G: nx.DiGraph, path: list[int]) -> gpd.GeoDataFrame:
    recs = []
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        recs.append({
            "u": int(u), "v": int(v),
            "mode": d["mode"],
            "length_m": float(d["length_m"]),
            "geometry": d["geometry"]
        })
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=f"EPSG:{NET_EPSG}")

def plot_route(route_edges: gpd.GeoDataFrame, out_png: Path):
    rp = route_edges.to_crs(PLOT_EPSG)
    styles = {
        "road":       dict(color="gray",   linewidth=2.0, alpha=0.9, zorder=2),
        "rail":       dict(color="blue",   linewidth=2.6, alpha=0.9, linestyle="--", zorder=3),
        "connection": dict(color="orange", linewidth=3.0, alpha=1.0, zorder=4),
        "unknown":    dict(color="lightgray", linewidth=1.5, alpha=0.6, zorder=1),
    }
    fig, ax = plt.subplots(figsize=(9,10))
    for m in ["road","rail","connection","unknown"]:
        g = rp[rp["mode"] == m]
        if len(g): g.plot(ax=ax, label=m, **styles[m])
    try:
        first, last = rp.geometry.iloc[0], rp.geometry.iloc[-1]
        x0, y0 = list(first.coords)[0]; x1, y1 = list(last.coords)[-1]
        ax.scatter([x0],[y0], s=40, color="green", zorder=10, label="origin")
        ax.scatter([x1],[y1], s=60, color="red", marker="X", zorder=11, label="destination")
    except Exception:
        pass
    minx, miny, maxx, maxy = rp.total_bounds
    padx, pady = (maxx-minx)*0.05, (maxy-miny)*0.05
    ax.set_xlim(minx-padx, maxx+padx); ax.set_ylim(miny-pady, maxy+pady)
    ax.set_axis_off()
    h,l = ax.get_legend_handles_labels(); uniq={}
    for hh,ll in zip(h,l):
        if ll and ll not in uniq: uniq[ll]=hh
    if uniq: ax.legend(uniq.values(), uniq.keys(), loc="lower left", frameon=False)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Plot saved: {out_png}")

# ---------------- Routing ----------------
def route_single_stage_distance(G: nx.DiGraph, o: int, d: int) -> list[int] | None:
    """Shortest path by 'length_m' (distance only)."""
    try:
        return nx.shortest_path(G, source=o, target=d, weight="length_m")
    except nx.NetworkXNoPath:
        return None

def ensure_interface_nodes(edges: gpd.GeoDataFrame) -> np.ndarray:
    """Nodes that touch BOTH access (road/connection) and rail edges."""
    acc  = edges[edges["mode"].isin(["road","connection"])]
    rail = edges[edges["mode"]=="rail"]
    acc_nodes  = np.unique(np.r_[acc["u"].to_numpy(dtype=int),  acc["v"].to_numpy(dtype=int)])  if len(acc)  else np.array([], int)
    rail_nodes = np.unique(np.r_[rail["u"].to_numpy(dtype=int), rail["v"].to_numpy(dtype=int)]) if len(rail) else np.array([], int)
    return np.intersect1d(acc_nodes, rail_nodes, assume_unique=False)

def route_forced_rail_distance(G: nx.DiGraph, id2pos: dict[int,tuple[float,float]],
                               edges: gpd.GeoDataFrame, o: int, d: int,
                               k_iface=INTERFACE_K_NEAREST) -> list[int] | None:
    """
    Three-stage forced rail, minimizing total distance:
      A) origin -> interface (access graph)
      B) interface -> interface (rail-only)
      C) interface -> destination (access graph)
    """
    access_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data["mode"] in ("road","connection")]
    rail_edges   = [(u, v, data) for u, v, data in G.edges(data=True) if data["mode"] == "rail"]
    if not rail_edges:
        return None

    G_access = nx.DiGraph(); G_access.add_edges_from(access_edges)
    G_rail   = nx.DiGraph(); G_rail.add_edges_from(rail_edges)

    iface_nodes = ensure_interface_nodes(edges)
    if len(iface_nodes) == 0:
        return None

    iface_pts = np.array([id2pos[i] for i in iface_nodes], float)
    iface_ids = np.array(iface_nodes, int)

    def k_nearest_iface(xy, k=k_iface):
        if KDTree is not None and len(iface_ids) > 0:
            tree = KDTree(iface_pts); _, idxs = tree.query(np.array(xy, float), k=min(k, len(iface_ids)))
            idxs = np.atleast_1d(idxs)
            return [int(iface_ids[int(i)]) for i in idxs]
        d2 = np.sum((iface_pts - np.array(xy, float))**2, axis=1)
        order = np.argsort(d2)[:min(k, len(d2))]
        return [int(iface_ids[int(i)]) for i in order]

    o_xy = id2pos[o]; d_xy = id2pos[d]
    cand_o = k_nearest_iface(o_xy);
    cand_d = k_nearest_iface(d_xy)

    def dijkstra_len(H, s, t):
        try:
            return nx.shortest_path(H, s, t, weight="length_m")
        except nx.NetworkXNoPath:
            return None

    # If o/d not in access, anchor to nearest access node
    def nearest_in_access(xy):
        ids = list(G_access.nodes())
        if not ids: return None
        pts = np.array([id2pos[i] for i in ids], float)
        if KDTree is not None and len(pts):
            tree = KDTree(pts); _, k = tree.query(np.array(xy,float))
            return int(ids[int(k)])
        d2 = np.sum((pts - np.array(xy,float))**2, axis=1)
        return int(ids[int(np.argmin(d2))])

    o_acc = o if o in G_access.nodes else nearest_in_access(o_xy)
    d_acc = d if d in G_access.nodes else nearest_in_access(d_xy)
    if o_acc is None or d_acc is None:
        return None

    bestL, bestP = None, None
    for rs in cand_o:
        if (rs not in G_access.nodes) or (rs not in G_rail.nodes):
            continue
        pA = dijkstra_len(G_access, o_acc, rs)
        if not pA:
            continue
        for rt in cand_d:
            if (rt not in G_access.nodes) or (rt not in G_rail.nodes):
                continue
            pB = dijkstra_len(G_rail, rs, rt)
            if not pB:
                continue
            pC = dijkstra_len(G_access, rt, d_acc)
            if not pC:
                continue

            def path_len(H, P):
                return sum(H[u][v]["length_m"] for u, v in zip(P[:-1], P[1:]))
            L = path_len(G_access, pA) + path_len(G_rail, pB) + path_len(G_access, pC)
            if bestL is None or L < bestL:
                bestL, bestP = L, (pA, pB, pC)

    if bestP is None:
        return None

    # Stitch pA + pB + pC (avoid duplicate joint)
    full = []
    for P in bestP:
        if not full:
            full.extend(P)
        else:
            if full[-1] == P[0]: full.extend(P[1:])
            else: full.extend(P)
    return full

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Distance-only multimodal routing on GPKG network.")
    ap.add_argument("--network", type=str, default=str(NETWORK_GPKG))

    # Chooser (like compare script)
    ap.add_argument("--orig", type=str, default="berlin",  help="Origin city name (built-in gazetteer)")
    ap.add_argument("--dest", type=str, default="frankfurt", help="Destination city name (built-in gazetteer)")
    ap.add_argument("--orig-xy", type=float, nargs=2, help="Origin coordinates: lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--dest-xy", type=float, nargs=2, help="Destination coordinates: lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--crs", type=int, default=4326, choices=[4326, 25832], help="CRS for --orig-xy/--dest-xy")

    # Names for filenames (optional; default to --orig/--dest labels)
    ap.add_argument("--orig-name", type=str, default="", help="Label used in filenames (defaults to --orig or 'origin')")
    ap.add_argument("--dest-name", type=str, default="", help="Label used in filenames (defaults to --dest or 'destination')")

    ap.add_argument("--snap-radius-m", type=float, default=SNAP_RADIUS_M)
    ap.add_argument("--force-rail", action="store_true", help="Force rail via interface nodes (still distance-based).")

    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help="Directory to write multimodal route files (GeoJSON + PNG).")
    ap.add_argument("--out-geojson", type=str, default="",
                    help="Explicit GeoJSON path (overrides --out-dir naming).")
    ap.add_argument("--out-png", type=str, default="",
                    help="Explicit PNG path (overrides --out-dir naming).")
    args = ap.parse_args()

    # 1) Load network
    edges, nodes = load_network(Path(args.network))
    G = build_graph_from_edges(edges)
    id2pos = id2pos_from_nodes(nodes)

    # 2) Resolve origin/destination XY via chooser
    # Priority: --orig-xy/--dest-xy → city names → (fallback error)
    if args.orig_xy and args.dest_xy:
        if args.crs == 4326:
            ox, oy = to_net_xy(args.orig_xy[0], args.orig_xy[1])
            dx, dy = to_net_xy(args.dest_xy[0], args.dest_xy[1])
        else:
            ox, oy = float(args.orig_xy[0]), float(args.orig_xy[1])
            dx, dy = float(args.dest_xy[0]), float(args.dest_xy[1])
        # Labels
        orig_label = args.orig_name or args.orig or "origin"
        dest_label = args.dest_name or args.dest or "destination"
    else:
        # city names → lon/lat → NET_EPSG
        def get_city_xy(name: str):
            ll = CITY_WGS84.get(name.strip().lower())
            if ll is None:
                print(f"ERROR: Unknown city '{name}'. Use --orig-xy/--dest-xy or extend gazetteer.", file=sys.stderr)
                sys.exit(2)
            return to_net_xy(ll[0], ll[1])
        ox, oy = get_city_xy(args.orig)
        dx, dy = get_city_xy(args.dest)
        orig_label = args.orig_name or args.orig
        dest_label = args.dest_name or args.dest

    # 3) Snap O/D to nearest network nodes
    o, odist = nearest_node_id(nodes, (ox, oy), max_radius=args.snap_radius_m)
    d, ddist = nearest_node_id(nodes, (dx, dy), max_radius=args.snap_radius_m)
    print(f"Snapped origin→node {o} ({odist:.1f} m), dest→node {d} ({ddist:.1f} m)")

    # 4) Single-stage distance
    path = route_single_stage_distance(G, o, d)
    used_rail = False
    if path:
        modes = [G[u][v]["mode"] for u, v in zip(path[:-1], path[1:])]
        used_rail = any(m == "rail" for m in modes)

    # 5) Optional forced-rail
    if args.force_rail and (path is None or not used_rail):
        print("ℹ️ Switching to 3-stage forced-rail (distance-based)…")
        path2 = route_forced_rail_distance(G, id2pos, edges, o, d)
        if path2:
            path = path2
        elif path is None:
            print("ERROR: Could not build any path.", file=sys.stderr); sys.exit(2)
        else:
            print("⚠️ Forced-rail failed; keeping single-stage result.")

    if path is None:
        print("ERROR: No path found between origin and destination.", file=sys.stderr)
        sys.exit(2)

    # 6) Build outputs (GeoJSON + PNG)
    route_edges = path_to_gdf(G, path)

    out_dir = Path(args.out_dir)
    o_slug = slugify(orig_label)
    d_slug = slugify(dest_label)

    if args.out_geojson:
        out_geo = Path(args.out_geojson)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_geo = out_dir / f"multimodal_{o_slug}_to_{d_slug}.geojson"  # matches EPS plotter default

    if args.out_png:
        out_png = Path(args.out_png)
    else:
        out_png = out_dir / f"multimodal_{o_slug}_to_{d_slug}.png"

    # Write GeoJSON in WGS84 (what plotter expects)
    route_edges.to_crs(epsg=4326).to_file(out_geo, driver="GeoJSON")
    print(f"✅ Route edges written: {out_geo}  (features: {len(route_edges)})")

    # Summary
    by_mode_km = (route_edges.groupby("mode")["length_m"].sum()/1000.0).sort_values(ascending=False)
    print("Lengths by mode (km):")
    for m, v in by_mode_km.items():
        print(f"  {m:11s}: {v:,.2f} km")
    print(f"Total length: {route_edges['length_m'].sum()/1000.0:,.2f} km")

    # Quick PNG preview
    plot_route(route_edges, out_png)

if __name__ == "__main__":
    main()
