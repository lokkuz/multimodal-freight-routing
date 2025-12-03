#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_routing_distance_compare.py

Route on ORIGINAL road and rail GeoJSONs (no merged multimodal network needed).
Builds *separate* road-only and rail-only graphs, finds shortest paths on each,
optionally plots, and writes TWO separate GeoJSON files with city names in filenames.

Examples:
  python 05_routing_distance_compare.py
  python 05_routing_distance_compare.py --metric time --speed-road 80 --speed-rail 160
  python 05_routing_distance_compare.py --orig Berlin --dest Frankfurt --plot graphs/berlin_frankfurt.png
  python 05_routing_distance_compare.py --orig-xy 13.405 52.52 --dest-xy 11.582 48.1351 --crs 4326
"""

from pathlib import Path
import argparse
import sys
import math
import json
import unicodedata

import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, LinearRing, MultiLineString

# -------- Defaults --------
BASE_DIR = Path(".").resolve()
DEF_ROADS = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
DEF_RAILS = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
DEF_OUTDIR = BASE_DIR / "graphs" / "routes"

NET_EPSG = 25832  # UTM32N (meters) for Germany

# Small gazetteer (lon, lat WGS84)
CITY_WGS84 = {
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

# ------------- Geo utils -------------
def _ensure_crs(gdf: gpd.GeoDataFrame, fallback_epsg=4326) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=fallback_epsg, allow_override=True)
    return gdf

def _to_net_crs(gdf: gpd.GeoDataFrame, epsg=NET_EPSG) -> gpd.GeoDataFrame:
    try:
        if gdf.crs and gdf.crs.to_epsg() == epsg:
            return gdf
    except Exception:
        pass
    return gdf.to_crs(epsg)

def _explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode MultiLineString to LineString; normalize LinearRing."""
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
    return gpd.GeoDataFrame(out, geometry="geometry", crs=g.crs).reset_index(drop=True)

def load_mode_gdf(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    g = _ensure_crs(g)
    g = _to_net_crs(g)
    g = _explode_lines(g)
    # keep only LineStrings with positive length
    g = g[g.geometry.type == "LineString"].copy()
    g = g[g.geometry.length > 0].reset_index(drop=True)
    g["length_m"] = g.geometry.length.astype(float)
    return g

# ------------- Graph building -------------
def build_graph_from_edges(gdf: gpd.GeoDataFrame, weight_col="length_m") -> nx.MultiGraph:
    """
    Build an undirected MultiGraph from LineStrings.
    Node IDs via a tiny grid-based merge (1 cm) to avoid float duplicates.
    """
    # Collect endpoints once for stable node ids
    coords = []
    for ls in gdf.geometry:
        a = ls.coords[0]; b = ls.coords[-1]
        coords.append((float(a[0]), float(a[1])))
        coords.append((float(b[0]), float(b[1])))
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)

    from math import floor, hypot
    tol = 0.01  # 1 cm in metric CRS
    cell = {}
    nodes_xy = []

    def key(xy): return (int(floor(xy[0] / tol)), int(floor(xy[1] / tol)))

    def get_id(xy):
        ix, iy = key(xy)
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

    G = nx.MultiGraph()
    # second pass: add edges
    for _, r in gdf.iterrows():
        ls: LineString = r.geometry
        a = (float(ls.coords[0][0]), float(ls.coords[0][1]))
        b = (float(ls.coords[-1][0]), float(ls.coords[-1][1]))
        u = get_id(a); v = get_id(b)
        w = float(r[weight_col])
        G.add_node(u, x=nodes_xy[u][0], y=nodes_xy[u][1])
        G.add_node(v, x=nodes_xy[v][0], y=nodes_xy[v][1])
        # store geometry and weight on the edge
        G.add_edge(u, v, weight=w, geom=ls)

    return G

# ------------- Routing helpers -------------
def nearest_graph_node_id(G: nx.Graph, x: float, y: float) -> int:
    """Return id of nearest node in G (Euclidean in network CRS)."""
    xs, ys, ids = [], [], []
    for nid, d in G.nodes(data=True):
        xs.append(d["x"]); ys.append(d["y"]); ids.append(nid)
    xs = np.asarray(xs); ys = np.asarray(ys); ids = np.asarray(ids, dtype=int)
    d2 = (xs - x) ** 2 + (ys - y) ** 2
    return int(ids[int(np.argmin(d2))])

def path_to_edges(G: nx.MultiGraph, path_nodes):
    """
    For a MultiGraph: for each (u,v) step in the node path, pick the parallel edge
    with minimal 'weight'. NetworkX 3.x compatible (uses get_edge_data).
    """
    out = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        data = G.get_edge_data(u, v)  # {key: attrdict, ...} or None
        if not data:
            # fallback for a simple Graph (no parallel edges)
            d = G.get_edge_data(u, v, default=None)
            if d is None:
                raise RuntimeError(f"No edge between {u} and {v}")
            out.append(d)
            continue
        best_d = None
        best_w = math.inf
        for k, d in data.items():
            w = d.get("weight", math.inf)
            if w < best_w:
                best_w = w
                best_d = d
        if best_d is None:
            raise RuntimeError(f"No weighted edge between {u} and {v}")
        out.append(best_d)
    return out

def run_route(G: nx.MultiGraph, sx: float, sy: float, tx: float, ty: float):
    s = nearest_graph_node_id(G, sx, sy)
    t = nearest_graph_node_id(G, tx, ty)
    try:
        path_nodes = nx.shortest_path(G, s, t, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, None
    segs = path_to_edges(G, path_nodes)
    total_w = sum(e["weight"] for e in segs)
    total_len = sum(e["geom"].length for e in segs if e.get("geom") is not None)
    return segs, (total_w, total_len)

# ------------- I/O helpers -------------
def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = "".join(c if c.isalnum() else "-" for c in s)
    s = "-".join(seg for seg in s.split("-") if seg)
    return s.lower()

def edges_to_gdf(edges, mode: str, crs_epsg: int):
    if not edges:
        return gpd.GeoDataFrame(columns=["mode", "geometry"], geometry="geometry", crs=f"EPSG:{crs_epsg}")
    geoms = [e["geom"] for e in edges]
    return gpd.GeoDataFrame({"mode": [mode]*len(geoms)}, geometry=geoms, crs=f"EPSG:{crs_epsg}")

def save_route_geojson(edges, mode: str, out_dir: Path, orig_label: str, dest_label: str, metric: str, src_epsg: int = NET_EPSG):
    """
    Save one route as a separate GeoJSON in WGS84 with a filename like:
      <mode>_<orig>_to_<dest>_<metric>.geojson
    """
    if not edges:
        print(f"{mode.upper()} | no path -> no GeoJSON written.")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{mode}_{slugify(orig_label)}_to_{slugify(dest_label)}_{metric}.geojson"
    out_path = out_dir / fname
    gdf = edges_to_gdf(edges, mode, crs_epsg=src_epsg).to_crs(4326)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"🗺️ wrote {mode} route: {out_path}")
    return out_path

def maybe_plot(out_png: Path, road_edges, rail_edges, ox, oy, dx, dy):
    if not out_png:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 10))
    # Road route
    plotted_road = False
    if road_edges:
        for e in road_edges:
            xs, ys = e["geom"].xy
            ax.plot(xs, ys, linewidth=2.0, alpha=0.9, label=None if plotted_road else "road route")
            plotted_road = True
    # Rail route
    plotted_rail = False
    if rail_edges:
        for e in rail_edges:
            xs, ys = e["geom"].xy
            ax.plot(xs, ys, linewidth=2.0, alpha=0.9, label=None if plotted_rail else "rail route")
            plotted_rail = True
    # Endpoints
    ax.scatter([ox], [oy], s=40)
    ax.scatter([dx], [dy], s=40)
    ax.set_aspect("equal")
    ax.set_title("Original networks: road vs rail")
    if plotted_road or plotted_rail:
        ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"🖼️ plot saved: {out_png}")

# ------------- Origin/destination parsing -------------
def to_net_xy(lon, lat):
    p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(NET_EPSG).iloc[0]
    return float(p.x), float(p.y)

def parse_city_or_xy(orig, dest, orig_xy, dest_xy, crs_code):
    if orig_xy:
        ox, oy = orig_xy
        if crs_code == 4326:
            ox, oy = to_net_xy(ox, oy)
    else:
        lon, lat = CITY_WGS84.get(orig.lower(), (None, None))
        if lon is None:
            raise ValueError(f"Unknown city '{orig}'. Use --orig-xy.")
        ox, oy = to_net_xy(lon, lat)

    if dest_xy:
        dx, dy = dest_xy
        if crs_code == 4326:
            dx, dy = to_net_xy(dx, dy)
    else:
        lon, lat = CITY_WGS84.get(dest.lower(), (None, None))
        if lon is None:
            raise ValueError(f"Unknown city '{dest}'. Use --dest-xy.")
        dx, dy = to_net_xy(lon, lat)

    return float(ox), float(oy), float(dx), float(dy)

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser(description="Shortest path on ORIGINAL road and rail GeoJSONs (separate graphs).")
    ap.add_argument("--roads", type=str, default=str(DEF_ROADS))
    ap.add_argument("--rails", type=str, default=str(DEF_RAILS))
    ap.add_argument("--metric", choices=["distance", "time"], default="distance")
    ap.add_argument("--speed-road", type=float, default=70.0, help="km/h if metric=time")
    ap.add_argument("--speed-rail", type=float, default=120.0, help="km/h if metric=time")
    ap.add_argument("--orig", type=str, default="berlin")
    ap.add_argument("--dest", type=str, default="frankfurt")
    ap.add_argument("--orig-xy", type=float, nargs=2, help="Origin as lon lat (WGS84) or x y with --crs 25832")
    ap.add_argument("--dest-xy", type=float, nargs=2, help="Destination as lon lat (WGS84) or x y with --crs 25832")
    ap.add_argument("--crs", type=int, default=4326, choices=[4326, 25832], help="CRS for --orig-xy/--dest-xy")
    ap.add_argument("--plot", type=str, default="", help="Optional PNG to save plot")
    ap.add_argument("--out-dir", type=str, default=str(DEF_OUTDIR), help="Directory to write separate route GeoJSONs")
    args = ap.parse_args()

    roads = load_mode_gdf(Path(args.roads))
    rails = load_mode_gdf(Path(args.rails))

    # prepare weights
    if args.metric == "time":
        # minutes = (meters/1000)/kmh * 60
        roads["weight"] = roads["length_m"] / 1000.0 / max(args.speed_road, 1e-6) * 60.0
        rails["weight"] = rails["length_m"] / 1000.0 / max(args.speed_rail, 1e-6) * 60.0
        metric_name = "time"
    else:
        roads["weight"] = roads["length_m"]
        rails["weight"] = rails["length_m"]
        metric_name = "distance"

    # build graphs
    print("Building road graph…")
    Groad = build_graph_from_edges(roads, weight_col="weight")
    print("Building rail graph…")
    Grail = build_graph_from_edges(rails, weight_col="weight")

    # origin/destination in network CRS
    try:
        ox, oy, dx, dy = parse_city_or_xy(args.orig, args.dest, args.orig_xy, args.dest_xy, args.crs)
    except ValueError as e:
        print("ERROR:", e, file=sys.stderr); sys.exit(2)

    # run routes
    print(f"Routing ROAD ({metric_name})…")
    road_edges, road_sum = run_route(Groad, ox, oy, dx, dy)
    print(f"Routing RAIL ({metric_name})…")
    rail_edges, rail_sum = run_route(Grail, ox, oy, dx, dy)

    # report
    if road_edges:
        print(f"ROAD  | {metric_name}: {road_sum[0]:.2f} | length_km: {road_sum[1]/1000:.2f} | edges: {len(road_edges)}")
    else:
        print("ROAD  | no path")

    if rail_edges:
        print(f"RAIL  | {metric_name}: {rail_sum[0]:.2f} | length_km: {rail_sum[1]/1000:.2f} | edges: {len(rail_edges)}")
    else:
        print("RAIL  | no path")

    # write separate GeoJSONs (WGS84) with city names in filenames
    out_dir = Path(args.out_dir)
    road_path = save_route_geojson(road_edges, "road", out_dir, args.orig, args.dest, metric_name, src_epsg=NET_EPSG)
    rail_path = save_route_geojson(rail_edges, "rail", out_dir, args.orig, args.dest, metric_name, src_epsg=NET_EPSG)

    # optional plot (network CRS)
    if args.plot:
        out_png = Path(args.plot)
        maybe_plot(out_png, road_edges, rail_edges, ox, oy, dx, dy)

if __name__ == "__main__":
    main()
