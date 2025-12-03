#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_routing_multiweight.py

Run multimodal routing for multiple weights on a merged network with explicit [u,v] nodes,
and SAVE the chosen weight values per edge into the output GeoJSON.

Inputs (GPKG):
  - layer 'network_edges': [u, v, mode, geometry, length_m, time_min, cost_eur, co2e_kg,
                            weight_time, weight_cost, weight_emission] (EPSG:25832)
  - layer 'network_nodes': [node_id, geometry] (EPSG:25832)

Weights supported:
  - distance  -> length_m
  - time      -> weight_time (fallback: time_min)
  - cost      -> weight_cost (fallback: cost_eur)
  - emission  -> weight_emission (fallback: co2e_kg)

For each weight this script:
  * computes a shortest path (optional 3-stage forced rail),
  * writes GeoJSON (WGS84) with per-edge weight columns,
  * writes a PNG preview colored by mode,
  * prints a brief per-mode length report.
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
NETWORK_GPKG = BASE / "graphs" / "multimodal_network_with_weights_sanitized.gpkg"
EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"
DEFAULT_OUT_DIR = BASE / "graphs" / "routes"

# ---------------- CRS ----------------
NET_EPSG  = 25832
PLOT_EPSG = 3857

# ---------------- Chooser defaults ----------------
CITY_WGS84 = {
    "berlin":     (13.4050, 52.5200),
    "cottbus":    (14.3343, 51.7607),
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

SNAP_RADIUS_M = 2000.0
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
    # Ensure length_m
    if "length_m" not in edges.columns:
        edges["length_m"] = edges.geometry.length.astype(float)
    else:
        mask = edges["length_m"].isna() | (edges["length_m"] <= 0)
        if mask.any():
            edges.loc[mask, "length_m"] = edges.loc[mask].geometry.length.astype(float)
    return edges, nodes

def build_graph_from_edges(edges: gpd.GeoDataFrame) -> nx.DiGraph:
    """Directed graph from explicit u/v. If duplicate (u,v), keep the smaller length_m."""
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        u = int(r["u"]); v = int(r["v"])
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        data = {k: r[k] for k in r.index if k not in ("u","v")}
        data["geometry"] = geom
        cur = G.get_edge_data(u, v)
        if (cur is None) or (float(r["length_m"]) < float(cur.get("length_m", np.inf))):
            G.add_edge(u, v, **data)
        curb = G.get_edge_data(v, u)
        if (curb is None) or (float(r["length_m"]) < float(curb.get("length_m", np.inf))):
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

# ---------------- Weight handling ----------------
WEIGHT_MAP = {
    "distance": ("length_m", None),
    "time":     ("weight_time", "time_min"),
    "cost":     ("weight_cost", "cost_eur"),
    "emission": ("weight_emission", "co2e_kg"),
}

def resolve_weight_attr(edges: gpd.GeoDataFrame, label: str) -> str | None:
    label = label.lower()
    if label not in WEIGHT_MAP: return None
    primary, fallback = WEIGHT_MAP[label]
    if primary in edges.columns: return primary
    if fallback and fallback in edges.columns: return fallback
    if label == "distance": return "length_m"
    return None

def stamp_edge_weight(H: nx.DiGraph, weight_col: str, default_len_fallback: bool = True):
    for _, _, d in H.edges(data=True):
        w = d.get(weight_col)
        if w is None and default_len_fallback:
            w = d.get("length_m", 1.0)
        d["_w"] = float(w)

# ---------------- Converters/plot ----------------
_KEEP_COLS = [
    "mode", "length_m",
    "time_min", "cost_eur", "co2e_kg",
    "weight_time", "weight_cost", "weight_emission"
]

def path_to_gdf(G: nx.DiGraph, path: list[int], weight_attr: str | None) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of route edges, keeping weight columns + 'weight_used' and 'cum_weight'."""
    recs = []
    cum = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        row = {"u": int(u), "v": int(v)}
        # copy known attributes if present
        for k in _KEEP_COLS:
            if k in d: row[k] = float(d[k]) if isinstance(d[k], (int,float,np.floating)) else d[k]
        row["mode"] = d.get("mode", row.get("mode", "unknown"))
        # geometry & fallback length
        row["geometry"] = d["geometry"]
        if "length_m" not in row:
            row["length_m"] = float(row["geometry"].length)
        # weight used this run
        wval = None
        if weight_attr and weight_attr in d:
            wval = float(d[weight_attr])
        elif weight_attr == "length_m":
            wval = float(row["length_m"])
        else:
            wval = float(d.get("length_m", row["geometry"].length))
        cum += wval
        row["weight_used"] = wval
        row["cum_weight"]  = cum
        recs.append(row)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=f"EPSG:{NET_EPSG}")

def plot_route(route_edges: gpd.GeoDataFrame, out_png: Path, title_suffix: str):
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
    if title_suffix:
        ax.set_title(title_suffix)
    h,l = ax.get_legend_handles_labels(); uniq={}
    for hh,ll in zip(h,l):
        if ll and ll not in uniq: uniq[ll]=hh
    if uniq: ax.legend(uniq.values(), uniq.keys(), loc="lower left", frameon=True, facecolor="white", framealpha=0.8)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Plot saved: {out_png}")

# ---------------- Interface nodes (forced rail) ----------------
def ensure_interface_nodes(edges: gpd.GeoDataFrame) -> np.ndarray:
    acc  = edges[edges["mode"].isin(["road","connection"])]
    rail = edges[edges["mode"]=="rail"]
    acc_nodes  = np.unique(np.r_[acc["u"].to_numpy(dtype=int),  acc["v"].to_numpy(dtype=int)])  if len(acc)  else np.array([], int)
    rail_nodes = np.unique(np.r_[rail["u"].to_numpy(dtype=int), rail["v"].to_numpy(dtype=int)]) if len(rail) else np.array([], int)
    return np.intersect1d(acc_nodes, rail_nodes, assume_unique=False)

def route_single_stage(G: nx.DiGraph, o: int, d: int) -> list[int] | None:
    try:
        return nx.shortest_path(G, source=o, target=d, weight="_w")
    except nx.NetworkXNoPath:
        return None

def route_forced_rail(G: nx.DiGraph, id2pos: dict, edges: gpd.GeoDataFrame,
                      o: int, d: int, k_iface=INTERFACE_K_NEAREST) -> list[int] | None:
    access_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data.get("mode") in ("road","connection")]
    rail_edges   = [(u, v, data) for u, v, data in G.edges(data=True) if data.get("mode") == "rail"]
    if not rail_edges: return None
    G_access = nx.DiGraph(); G_access.add_edges_from(access_edges)
    G_rail   = nx.DiGraph(); G_rail.add_edges_from(rail_edges)
    # weights already stamped on G; copy to subgraphs too
    for H in (G_access, G_rail):
        for u, v, dta in H.edges(data=True):
            dta["_w"] = G[u][v]["_w"]

    iface_nodes = ensure_interface_nodes(edges)
    if len(iface_nodes) == 0: return None

    iface_pts = np.array([id2pos[i] for i in iface_nodes], float)
    iface_ids = np.array(iface_nodes, int)

    def k_nearest_iface(xy, k=INTERFACE_K_NEAREST):
        if KDTree is not None and len(iface_ids) > 0:
            tree = KDTree(iface_pts); _, idxs = tree.query(np.array(xy,float), k=min(k, len(iface_ids)))
            idxs = np.atleast_1d(idxs)
            return [int(iface_ids[int(i)]) for i in idxs]
        d2 = np.sum((iface_pts - np.array(xy,float))**2, axis=1)
        order = np.argsort(d2)[:min(k, len(d2))]
        return [int(iface_ids[int(i)]) for i in order]

    o_xy = id2pos[o]; d_xy = id2pos[d]
    cand_o = k_nearest_iface(o_xy);
    cand_d = k_nearest_iface(d_xy)

    def dijkstra(H, s, t):
        try: return nx.shortest_path(H, s, t, weight="_w")
        except nx.NetworkXNoPath: return None

    # If o/d not in access
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
    if o_acc is None or d_acc is None: return None

    bestW, bestP = None, None
    for rs in cand_o:
        if (rs not in G_access.nodes) or (rs not in G_rail.nodes): continue
        pA = dijkstra(G_access, o_acc, rs)
        if not pA: continue
        for rt in cand_d:
            if (rt not in G_access.nodes) or (rt not in G_rail.nodes): continue
            pB = dijkstra(G_rail, rs, rt)
            if not pB: continue
            pC = dijkstra(G_access, rt, d_acc)
            if not pC: continue

            def path_weight(H, P): return sum(H[u][v]["_w"] for u, v in zip(P[:-1], P[1:]))
            W = path_weight(G_access, pA) + path_weight(G_rail, pB) + path_weight(G_access, pC)
            if bestW is None or W < bestW:
                bestW, bestP = W, (pA, pB, pC)

    if bestP is None: return None
    full = []
    for P in bestP:
        if not full: full.extend(P)
        else:
            if full[-1] == P[0]: full.extend(P[1:])
            else: full.extend(P)
    return full

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Multimodal routing for multiple weights (and save weights per route edge).")
    ap.add_argument("--network", type=str, default=str(NETWORK_GPKG))

    # Chooser
    ap.add_argument("--orig", type=str, default="Berlin",  help="Origin city name (built-in gazetteer)")
    ap.add_argument("--dest", type=str, default="Cottbus",   help="Destination city name (built-in gazetteer)")
    ap.add_argument("--orig-xy", type=float, nargs=2, help="Origin lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--dest-xy", type=float, nargs=2, help="Destination lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--crs", type=int, default=4326, choices=[4326, 25832], help="CRS for --orig-xy/--dest-xy")

    ap.add_argument("--snap-radius-m", type=float, default=SNAP_RADIUS_M)
    ap.add_argument("--force-rail", action="store_true", help="If single-stage path has no rail, try 3-stage forced rail.")
    ap.add_argument("--k-near", type=int, default=INTERFACE_K_NEAREST, help="Candidate interface nodes near O/D (forced rail).")

    # Which weights to run
    ap.add_argument("--weights", type=str, default="distance,time,cost,emission",
                    help="Comma list of weights to run (distance,time,cost,emission)")

    # Outputs
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help="Directory to write route files (GeoJSON + PNG) per weight.")
    args = ap.parse_args()

    # Load network
    edges, nodes = load_network(Path(args.network))
    G = build_graph_from_edges(edges)
    id2pos = id2pos_from_nodes(nodes)

    # Resolve O/D
    if args.orig_xy and args.dest_xy:
        if args.crs == 4326:
            ox, oy = to_net_xy(args.orig_xy[0], args.orig_xy[1])
            dx, dy = to_net_xy(args.dest_xy[0], args.dest_xy[1])
        else:
            ox, oy = float(args.orig_xy[0]), float(args.orig_xy[1])
            dx, dy = float(args.dest_xy[0]), float(args.dest_xy[1])
        orig_label, dest_label = args.orig, args.dest
    else:
        def city_xy(name: str):
            ll = CITY_WGS84.get(name.strip().lower())
            if ll is None:
                print(f"ERROR: Unknown city '{name}'. Use --orig-xy/--dest-xy or extend gazetteer.", file=sys.stderr)
                sys.exit(2)
            return to_net_xy(ll[0], ll[1])
        ox, oy = city_xy(args.orig)
        dx, dy = city_xy(args.dest)
        orig_label, dest_label = args.orig, args.dest

    # Snap
    o, odist = nearest_node_id(nodes, (ox, oy), max_radius=args.snap_radius_m)
    d, ddist = nearest_node_id(nodes, (dx, dy), max_radius=args.snap_radius_m)
    print(f"Snapped origin→node {o} ({odist:.1f} m), dest→node {d} ({ddist:.1f} m)")

    # Prepare outputs
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    o_slug, d_slug = slugify(orig_label), slugify(dest_label)

    wanted = [w.strip().lower() for w in args.weights.split(",") if w.strip()]
    wanted = [w for w in wanted if w in WEIGHT_MAP] or ["distance"]

    for wlabel in wanted:
        weight_col = resolve_weight_attr(edges, wlabel)
        if weight_col is None:
            print(f"⚠️  Skipping '{wlabel}': required columns not found on edges.")
            continue

        print(f"\n=== Routing by {wlabel} (edge attribute '{weight_col}') ===")
        # Stamp weights
        stamp_edge_weight(G, weight_col)

        # Single-stage
        path = route_single_stage(G, o, d)
        used_rail = False
        if path:
            modes = [G[u][v].get("mode","unknown") for u, v in zip(path[:-1], path[1:])]
            used_rail = any(m == "rail" for m in modes)

        # Optional forced rail
        if args.force_rail and (path is None or not used_rail):
            print("ℹ️  Single-stage path had no rail or failed. Trying 3-stage forced rail…")
            p2 = route_forced_rail(G, id2pos, edges, o, d, k_iface=args.k_near)
            if p2: path = p2
            elif path is None:
                print("❌ Could not build any path for this weight.", file=sys.stderr); continue
            else:
                print("⚠️ Forced-rail failed; keeping single-stage result.")

        if path is None:
            print("❌ No path found for this weight.")
            continue

        # Build outputs (include weights per edge)
        route_edges = path_to_gdf(G, path, weight_attr=weight_col)

        # Save GeoJSON (WGS84), now WITH weight columns
        out_geo = out_dir / f"multimodal_{o_slug}_to_{d_slug}_{wlabel}.geojson"
        out_png = out_dir / f"multimodal_{o_slug}_to_{d_slug}_{wlabel}.png"
        route_edges.to_crs(epsg=4326).to_file(out_geo, driver="GeoJSON")
        print(f"✅ Route ({wlabel}) written: {out_geo}  (features: {len(route_edges)})")

        # Per-mode report
        by_mode = route_edges.groupby("mode")["length_m"].sum()
        total_len_km = by_mode.sum() / 1000.0
        print("  Lengths by mode (km):")
        for m, meters in by_mode.sort_values(ascending=False).items():
            print(f"    {m:11s}: {meters/1000.0:,.2f} km")
        # Total weight
        total_weight = float(route_edges["weight_used"].sum())
        unit = {
            "distance": "m",
            "time": "min",
            "cost": "€",
            "emission": "kg CO2e",
        }.get(wlabel, "units")
        print(f"  Total {wlabel}: {total_weight:,.3f} {unit}")
        print(f"  Total length: {total_len_km:,.2f} km")

        # PNG preview
        plot_route(route_edges, out_png, title_suffix=f"{orig_label} → {dest_label} [{wlabel}]")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
