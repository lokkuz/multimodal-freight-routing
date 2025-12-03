#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_plot_two_routes_eps.py

Plot TWO multimodal routes in a single figure against the Germany border.
Each route is drawn by 'mode' (road/rail/connection) with distinct linestyles:
  - Route A: solid
  - Route B: dashed

Inputs:
  * Either pass explicit files with --route-a / --route-b
  * Or let it derive default paths from --orig/--dest and --weight-a/--weight-b:
      graphs/routes/multimodal_<orig>_to_<dest>_<weight>.geojson

Extras:
  * Optional network underlay (clipped around the route union)
  * Fixed-frame controls (explicit bbox, or from reference route), or close-up
  * City markers and legend styling
  * Prints length summaries per route (and optional CSVs)

Examples
--------
# Compare distance vs time for Cottbus→Berlin in one plot, close-up frame
python 06_plot_two_routes_eps.py \
  --orig Cottbus --dest Berlin \
  --weight-a distance --weight-b time \
  --zoom-route-km 12 \
  --out graphs/plots/cottbus_berlin_compare_distance_time

# Use explicit files and fixed frame from a reference route + padding, add network
python 06_plot_two_routes_eps.py \
  --route-a graphs/routes/multimodal_cottbus_to_berlin_distance.geojson \
  --route-b graphs/routes/multimodal_cottbus_to_berlin_cost.geojson \
  --frame-from graphs/routes/multimodal_cottbus_to_berlin_distance.geojson \
  --frame-pad-km 8 \
  --network graphs/multimodal_network_with_weights_sanitized.gpkg --network-layer network_edges
"""

from pathlib import Path
import argparse
import unicodedata
import warnings

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, LinearRing
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------- Paths & CRS ----------------
BASE_DIR = Path(".").resolve()
DEF_BORDER    = BASE_DIR / "input_files" / "germany_state_borders.geojson"
DEF_ROUTESDIR = BASE_DIR / "graphs" / "routes"
DEF_PLOTSDIR  = BASE_DIR / "graphs" / "plots"

PLOT_EPSG = 25832
NET_EPSG  = 25832

# ---------------- STYLE DEFAULTS ----------------
DEFAULT_LABEL_FONTFAMILY  = "Arial"
DEFAULT_LABEL_FONTSIZE    = 24
DEFAULT_MARKER_SIZE       = 125

DEFAULT_LEGEND_FONTFAMILY = "Arial"
DEFAULT_LEGEND_FONTSIZE   = 24
DEFAULT_LEGEND_LOC        = "upper right"
DEFAULT_LEGEND_ALPHA      = 1.00

DEFAULT_EXTRA_LEFT_FRAC   = 0.00
DEFAULT_EXTRA_RIGHT_FRAC  = 0.20

# Colors for modes (consistent across both routes)
MODE_COLORS = {
    "road":       "darkblue",
    "rail":       "darkred",
    "connection": "goldenrod",
    "unknown":    "#16a34a",
}

# ---------------- Gazetteer (lon, lat WGS84) ----------------
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
    "nürnberg":   (11.0796, 49.4521),
}

# ---------------- Helpers ----------------
def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = "".join(c if c.isalnum() else "-" for c in s)
    s = "-".join(seg for seg in s.split("-") if seg)
    return s.lower()

def _ensure_crs(gdf: gpd.GeoDataFrame, fallback_epsg=4326) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    if gdf.crs is None:
        warnings.warn(f"Input had no CRS; assuming EPSG:{fallback_epsg}.")
        gdf = gdf.set_crs(epsg=fallback_epsg, allow_override=True)
    return gdf

def _to_plot_crs(gdf: gpd.GeoDataFrame, epsg=PLOT_EPSG) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    try:
        if gdf.crs and gdf.crs.to_epsg() == epsg:
            return gdf
    except Exception:
        pass
    return gdf.to_crs(epsg)

def _explode_lines_keep_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or len(gdf) == 0:
        return gdf
    g = gdf.explode(index_parts=False).reset_index(drop=True)
    def _norm(geom):
        if isinstance(geom, LinearRing):
            return LineString(geom.coords)
        return geom
    g["geometry"] = g["geometry"].apply(_norm)
    return g

def load_route(path: Path):
    if not path or not path.exists():
        return None
    g = gpd.read_file(path)
    g = _ensure_crs(g)
    g = _explode_lines_keep_lines(g)
    g = _to_plot_crs(g)
    g = g[g.geometry.geom_type.isin(["LineString"])].copy()
    if "mode" in g.columns:
        g["mode"] = g["mode"].astype(str).str.lower()
    else:
        g["mode"] = "unknown"
    g = g.reset_index(drop=True)
    return g

def load_border(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    g = _ensure_crs(g)
    g = _to_plot_crs(g)
    return g

def route_default_path(orig: str, dest: str, weight: str, routes_dir: Path) -> Path:
    o = slugify(orig); d = slugify(dest); w = slugify(weight)
    return routes_dir / f"multimodal_{o}_to_{d}_{w}.geojson"

def to_plot_xy_from_lonlat(lon, lat):
    gs = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(PLOT_EPSG)
    p = gs.iloc[0]; return float(p.x), float(p.y)

def to_plot_xy_from_net(x, y):
    gs = gpd.GeoSeries([Point(x, y)], crs=NET_EPSG).to_crs(PLOT_EPSG)
    p = gs.iloc[0]; return float(p.x), float(p.y)

def resolve_marker_points(args):
    orig_label = args.orig; dest_label = args.dest
    if args.orig_xy and args.dest_xy:
        ox, oy = float(args.orig_xy[0]), float(args.orig_xy[1])
        dx, dy = float(args.dest_xy[0]), float(args.dest_xy[1])
        if args.crs == 4326:
            o_xy = to_plot_xy_from_lonlat(ox, oy); d_xy = to_plot_xy_from_lonlat(dx, dy)
        else:
            o_xy = to_plot_xy_from_net(ox, oy); d_xy = to_plot_xy_from_net(dx, dy)
        return o_xy, d_xy, orig_label, dest_label
    lo = CITY_WGS84.get(args.orig.strip().lower())
    ld = CITY_WGS84.get(args.dest.strip().lower())
    if lo and ld:
        return to_plot_xy_from_lonlat(*lo), to_plot_xy_from_lonlat(*ld), orig_label, dest_label
    warnings.warn("City coordinates not resolved.")
    return None, None, orig_label, dest_label

def compute_extent(bounds_list, pad_ratio=0.05):
    finite_bounds = [b for b in bounds_list if b is not None]
    if not finite_bounds:
        return None
    minx = min(b[0] for b in finite_bounds)
    miny = min(b[1] for b in finite_bounds)
    maxx = max(b[2] for b in finite_bounds)
    maxy = max(b[3] for b in finite_bounds)
    dx = maxx - minx; dy = maxy - miny
    if dx == 0 or dy == 0:
        pad = 10000.0
        return (minx - pad, miny - pad, maxx + pad, maxy + pad)
    px = dx * pad_ratio; py = dy * pad_ratio
    return (minx - px, miny - py, maxx + px, maxy + py)

def route_closeup_extent_union(route_a, route_b, pad_km: float):
    """Close-up bbox around the UNION of two routes, padded by pad_km."""
    pieces = []
    for g in (route_a, route_b):
        if g is None or len(g) == 0:
            continue
        pieces.append(g.unary_union)
    if not pieces:
        return None
    union_geom = gpd.GeoSeries(pieces, crs=PLOT_EPSG).unary_union
    minx, miny, maxx, maxy = gpd.GeoSeries([union_geom], crs=PLOT_EPSG).total_bounds
    pad = pad_km * 1000.0
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)

def _bbox_to_plot_epsg(bbox, bbox_crs=25832):
    if bbox is None:
        return None
    minx, miny, maxx, maxy = map(float, bbox)
    if int(bbox_crs) == PLOT_EPSG:
        return (minx, miny, maxx, maxy)
    g = gpd.GeoSeries([Point(minx, miny), Point(maxx, maxy)], crs=int(bbox_crs)).to_crs(PLOT_EPSG)
    p0, p1 = g.iloc[0], g.iloc[1]
    return (float(p0.x), float(p0.y), float(p1.x), float(p1.y))

def frame_from_route_file(route_path: Path, pad_km: float):
    if not route_path.exists():
        return None
    g = gpd.read_file(route_path)
    if g.crs is None:
        g = g.set_crs(epsg=PLOT_EPSG, allow_override=True)
    if g.crs.to_epsg() != PLOT_EPSG:
        g = g.to_crs(PLOT_EPSG)
    g = g[g.geometry.notna()]
    if g.empty:
        return None
    minx, miny, maxx, maxy = g.total_bounds
    pad = pad_km * 1000.0
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)

# ---------- Network underlay (optional) ----------
def load_network_edges(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    if not path or not path.exists():
        return None
    if path.suffix.lower() in {".gpkg", ".geopackage"}:
        g = gpd.read_file(path, layer=layer or "network_edges")
    else:
        g = gpd.read_file(path)
    g = _ensure_crs(g, fallback_epsg=PLOT_EPSG)
    if g.crs.to_epsg() != PLOT_EPSG:
        g = g.to_crs(PLOT_EPSG)
    g = _explode_lines_keep_lines(g)
    g = g[g.geometry.geom_type.isin(["LineString"])].copy()
    if "mode" in g.columns:
        g["mode"] = g["mode"].astype(str).str.lower()
    else:
        g["mode"] = "unknown"
    return g

def clip_network_to_bbox(net: gpd.GeoDataFrame, bbox) -> gpd.GeoDataFrame:
    if net is None or len(net) == 0 or bbox is None:
        return net
    minx, miny, maxx, maxy = bbox
    # quick spatial filter
    box = gpd.GeoSeries([Point(minx, miny), Point(maxx, maxy)], crs=PLOT_EPSG).unary_union.envelope
    cand_idx = list(net.sindex.query(box, predicate="intersects"))
    if not cand_idx:
        return net.iloc[0:0].copy()
    sub = net.iloc[cand_idx].copy()
    return sub

def clip_network_to_route(network_gdf: gpd.GeoDataFrame, route_gdf: gpd.GeoDataFrame,
                          buffer_km: float) -> gpd.GeoDataFrame:
    if network_gdf is None or len(network_gdf) == 0:
        return None
    if route_gdf is None or len(route_gdf) == 0 or buffer_km <= 0:
        return network_gdf
    buf = route_gdf.unary_union.buffer(buffer_km * 1000.0)
    # fast spatial index filter
    cand_idx = list(network_gdf.sindex.query(buf, predicate="intersects"))
    if not cand_idx:
        return network_gdf.iloc[0:0].copy()
    cand = network_gdf.iloc[cand_idx].copy()
    # precise clip
    cand = cand[cand.intersects(buf)]
    return cand

def maybe_simplify_and_sample(net: gpd.GeoDataFrame, simplify_m: float, max_edges: int) -> gpd.GeoDataFrame:
    if net is None or len(net) == 0:
        return net
    g = net
    if simplify_m and simplify_m > 0:
        g = g.copy()
        g["geometry"] = g.geometry.simplify(simplify_m, preserve_topology=True)
    if len(g) > max_edges:
        g = g.sample(n=max_edges, random_state=42)
    return g

def plot_network_underlay(ax, net_gdf: gpd.GeoDataFrame, lw: float, alpha: float):
    if net_gdf is None or len(net_gdf) == 0:
        return
    colors = {
        "road":       "#9aa4ad",
        "rail":       "#c78b8b",
        "connection": "#bfa76b",
        "unknown":    "#b3b3b3",
    }
    for m in ["road","rail","connection","unknown"]:
        g = net_gdf[net_gdf["mode"] == m]
        if len(g):
            g.plot(ax=ax, linewidth=lw, color=colors.get(m, "#b3b3b3"), alpha=alpha, zorder=2)

# ---------------- Summaries ----------------
def length_summary(route_gdf: gpd.GeoDataFrame, label: str):
    if route_gdf is None or len(route_gdf) == 0:
        print(f"(empty) {label}")
        return
    g = route_gdf
    if g.crs and g.crs.to_epsg() != PLOT_EPSG:
        g = g.to_crs(PLOT_EPSG)
    seg = pd.DataFrame({
        "mode": g["mode"].astype(str).str.lower() if "mode" in g.columns else "multimodal",
        "length_m": g.length.astype(float).values
    })
    seg["length_km"] = seg["length_m"]/1000.0
    summary = seg.groupby("mode", as_index=False).agg(
        segments=("length_m","size"),
        total_length_km=("length_km","sum")
    )
    print(f"\n=== Length summary: {label} ===")
    print(f"Total: {seg['length_km'].sum():,.2f} km across {len(seg)} segments")
    for _, r in summary.sort_values("mode").iterrows():
        print(f"  {r['mode']:>11s}: {r['total_length_km']:,.2f} km | segments: {int(r['segments'])}")
    print("=================================\n")

# ---------------- Plot ----------------
def plot_two_routes(multi_a, multi_b, border_gdf, net_underlay, *,
                    out_eps: Path, out_png: Path, dpi: int, title: str,
                    orig_marker, dest_marker, orig_label: str, dest_label: str,
                    label_fontfamily: str, label_fontsize: int, marker_size: int,
                    legend_fontfamily: str, legend_fontsize: int, legend_loc: str,
                    legend_alpha: float, extra_left_frac: float, extra_right_frac: float,
                    extent_override=None, network_lw=0.4, network_alpha=0.25):
    fig, ax = plt.subplots(figsize=(9, 10))

    # Border
    if border_gdf is not None and len(border_gdf):
        border_gdf.plot(ax=ax, facecolor="none", edgecolor="dimgrey", linewidth=0.8, zorder=1)

    # Network underlay
    plot_network_underlay(ax, net_underlay, lw=network_lw, alpha=network_alpha)

    # Styles per route
    styleA = dict(linewidth=3, linestyle="-", color="green")
    styleB = dict(linewidth=3, linestyle="-", color="#0000D2")

    def _plot_by_mode(gdf, label_prefix, style):
        if gdf is None or len(gdf) == 0: return
        for mode in ["road","rail","connection","unknown"]:
            gg = gdf[gdf["mode"]==mode] if "mode" in gdf.columns else gdf.iloc[0:0]
            if len(gg):
                gg.plot(ax=ax, zorder=6, label=f"{label_prefix}: {mode}", **style)

    _plot_by_mode(multi_a, "Time", styleA)
    _plot_by_mode(multi_b, "Cost", styleB)

    # Extent
    if extent_override:
        minx, miny, maxx, maxy = extent_override
        width = maxx - minx
        ax.set_xlim(minx - extra_left_frac * width, maxx + extra_right_frac * width)
        ax.set_ylim(miny, maxy)
    else:
        b_a = tuple(multi_a.total_bounds) if multi_a is not None and len(multi_a) else None
        b_b = tuple(multi_b.total_bounds) if multi_b is not None and len(multi_b) else None
        b_border = tuple(border_gdf.total_bounds) if border_gdf is not None and len(border_gdf) else None
        extent = compute_extent([b_border, b_a, b_b], pad_ratio=0.06)
        if extent:
            width = extent[2] - extent[0]
            ax.set_xlim(extent[0] - extra_left_frac * width, extent[2] + extra_right_frac * width)
            ax.set_ylim(extent[1], extent[3])

    # Markers & labels
    if orig_marker:
        ox, oy = orig_marker
        ax.scatter([ox],[oy], s=marker_size, color="#10b981", edgecolor="black", linewidth=0.5, zorder=10, label="Origin")
        ax.annotate(orig_label, (ox, oy), xytext=(8,8), textcoords="offset points",
                    fontsize=label_fontsize, fontfamily=label_fontfamily,
                    weight="bold", zorder=11,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    if dest_marker:
        dx, dy = dest_marker
        ax.scatter([dx],[dy], s=marker_size*1.1, color="#ef4444", edgecolor="black", linewidth=0.5,
                   marker="D", zorder=10, label="Destination")
        ax.annotate(dest_label, (dx, dy), xytext=(8,-10), textcoords="offset points",
                    fontsize=label_fontsize, fontfamily=label_fontfamily,
                    weight="bold", zorder=11,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

    ax.set_aspect("equal"); ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc=legend_loc,
                  frameon=True, facecolor="white", framealpha=float(legend_alpha),
                  prop={"family": legend_fontfamily, "size": legend_fontsize})

    out_eps.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_eps, format="eps", bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ EPS saved: {out_eps}")
    print(f"✅ PNG saved: {out_png}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot TWO multimodal routes in one figure (by mode, solid vs dashed).")
    ap.add_argument("--orig", type=str, default="Berlin", help="Origin city label")
    ap.add_argument("--dest", type=str, default="Cottbus", help="Destination city label")

    # Either give files or weights (derived from routes dir)
    ap.add_argument("--route-a", type=str, default="", help="Explicit path to route A")
    ap.add_argument("--route-b", type=str, default="", help="Explicit path to route B")
    ap.add_argument("--weight-a", type=str, default="time", help="Weight label for route A if file not given")
    ap.add_argument("--weight-b", type=str, default="cost",      help="Weight label for route B if file not given")
    ap.add_argument("--routes-dir", type=str, default=str(DEF_ROUTESDIR))

    ap.add_argument("--border", type=str, default=str(DEF_BORDER))
    ap.add_argument("--out", type=str, default="", help="Output base path (no extension). Default builds from orig/dest/weights.")
    ap.add_argument("--dpi", type=int, default=220)

    # City markers
    ap.add_argument("--orig-xy", type=float, nargs=2)
    ap.add_argument("--dest-xy", type=float, nargs=2)
    ap.add_argument("--crs", type=int, default=4326, choices=[4326, 25832])

    # Fonts / legend / padding
    ap.add_argument("--label-font", type=str, default=DEFAULT_LABEL_FONTFAMILY)
    ap.add_argument("--label-fontsize", type=int, default=DEFAULT_LABEL_FONTSIZE)
    ap.add_argument("--marker-size", type=int, default=DEFAULT_MARKER_SIZE)

    ap.add_argument("--legend-font", type=str, default=DEFAULT_LEGEND_FONTFAMILY)
    ap.add_argument("--legend-fontsize", type=int, default=DEFAULT_LEGEND_FONTSIZE)
    ap.add_argument("--legend-loc", type=str, default=DEFAULT_LEGEND_LOC)
    ap.add_argument("--legend-alpha", type=float, default=DEFAULT_LEGEND_ALPHA)

    ap.add_argument("--extra-left",  type=float, default=DEFAULT_EXTRA_LEFT_FRAC)
    ap.add_argument("--extra-right", type=float, default=DEFAULT_EXTRA_RIGHT_FRAC)

    # Fixed frame controls
    ap.add_argument("--frame-bbox", type=float, nargs=4, metavar=("MINX","MINY","MAXX","MAXY"),
                    help="Fixed frame bbox (EPSG:25832 by default; set --frame-crs 4326 for lon/lat)")
    ap.add_argument("--frame-crs", type=int, default=25832, choices=[25832, 4326])
    ap.add_argument("--frame-from", type=str, default="graphs/routes/multimodal_berlin_to_cottbus_emission.geojson",
                    help="Use a reference route file to define frame")
    ap.add_argument("--frame-pad-km", type=float, default=5.0)

    # Close-up (if no fixed frame given)
    ap.add_argument("--zoom-route-km", type=float, default=15.0,
                    help="If >0, zoom to union of both routes, padded by this many km")

    # Network underlay (optional)
    ap.add_argument("--network", type=str, default="graphs/multimodal_network_with_weights_sanitized.gpkg", help="Path to network edges (GPKG/GeoJSON)")
    ap.add_argument("--network-layer", type=str, default="network_edges")
    ap.add_argument("--network-modes", type=str, default="road,rail,connection")
    ap.add_argument("--network-alpha", type=float, default=0.7)
    ap.add_argument("--network-lw", type=float, default=0.5)
    ap.add_argument("--network-simplify-m", type=float, default=0.0)
    ap.add_argument("--network-max-edges", type=int, default=250_000)
    ap.add_argument("--network-buffer-km", type=float, default=100.0,
                    help="Clip network to this buffer around route (km). Use 0 to skip clipping")

    args = ap.parse_args()

    routes_dir = Path(args.routes_dir)
    border_gdf = load_border(Path(args.border))

    # Determine files
    if args.route_a:
        pa = Path(args.route_a)
    else:
        pa = route_default_path(args.orig, args.dest, args.weight_a, routes_dir)
    if args.route_b:
        pb = Path(args.route_b)
    else:
        pb = route_default_path(args.orig, args.dest, args.weight_b, routes_dir)

    print(f"Route A: {pa} ({'OK' if pa.exists() else 'MISSING'})")
    print(f"Route B: {pb} ({'OK' if pb.exists() else 'MISSING'})")

    route_a = load_route(pa) if pa.exists() else None
    route_b = load_route(pb) if pb.exists() else None
    if (route_a is None or len(route_a)==0) and (route_b is None or len(route_b)==0):
        raise SystemExit("Both routes missing/empty.")

    # City markers
    o_xy, d_xy, o_label, d_label = resolve_marker_points(args)

    # --- Load/prepare network underlay once per weight (optional) ---
    net_underlay = None
    if args.network:
        net_all = load_network_edges(Path(args.network), layer=args.network_layer)
        if net_all is not None and len(net_all):
            # filter modes
            wanted = {m.strip().lower() for m in args.network_modes.split(",") if m.strip()}
            if "mode" in net_all.columns and wanted:
                net_all = net_all[net_all["mode"].isin(wanted)].copy()
            # clip to route buffer
            net_clip = clip_network_to_route(net_all, route_a, buffer_km=float(args.network_buffer_km))
            # optional simplify & sample for speed
            net_underlay = maybe_simplify_and_sample(net_clip, args.network_simplify_m, args.network_max_edges)

    # Frame selection (precedence)
    extent_override = None
    if args.frame_bbox is not None:
        extent_override = _bbox_to_plot_epsg(args.frame_bbox, bbox_crs=args.frame_crs)
    elif args.frame_from:
        ref_bbox = frame_from_route_file(Path(args.frame_from), pad_km=float(args.frame_pad_km))
        if ref_bbox:
            extent_override = ref_bbox
    elif args.zoom_route_km and args.zoom_route_km > 0:
        extent_override = route_closeup_extent_union(route_a, route_b, args.zoom_route_km)

    # Output paths
    if args.out:
        base = Path(args.out)
        out_eps = base.with_suffix(".eps")
        out_png = base.with_suffix(".png")
    else:
        o_slug, d_slug = slugify(args.orig), slugify(args.dest)
        wA, wB = (args.weight_a if not args.route_a else pa.stem.split("_")[-1],
                  args.weight_b if not args.route_b else pb.stem.split("_")[-1])
        out_eps = DEF_PLOTSDIR / f"{o_slug}_to_{d_slug}_compare_{wA}_{wB}.eps"
        out_png = out_eps.with_suffix(".png")

    # Print summaries
    if route_a is not None and len(route_a): length_summary(route_a, f"A [{args.weight_a if not args.route_a else pa.name}]")
    if route_b is not None and len(route_b): length_summary(route_b, f"B [{args.weight_b if not args.route_b else pb.name}]")

    # Plot
    plot_two_routes(
        route_a, route_b, border_gdf, net_underlay,
        out_eps=out_eps, out_png=out_png, dpi=args.dpi, title=f"{args.orig} → {args.dest}",
        orig_marker=o_xy, dest_marker=d_xy, orig_label=o_label, dest_label=d_label,
        label_fontfamily=args.label_font, label_fontsize=args.label_fontsize, marker_size=args.marker_size,
        legend_fontfamily=args.legend_font, legend_fontsize=args.legend_fontsize,
        legend_loc=args.legend_loc, legend_alpha=args.legend_alpha,
        extra_left_frac=args.extra_left, extra_right_frac=args.extra_right,
        extent_override=extent_override,
        network_lw=args.network_lw, network_alpha=args.network_alpha
    )

    print("Done.")

if __name__ == "__main__":
    main()
