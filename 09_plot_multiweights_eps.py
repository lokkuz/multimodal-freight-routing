#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_plot_multiweights_eps.py

Plot ONLY the multimodal route (no road/rail comparison) against the Germany border,
for one or multiple weights (distance/time/cost/emission). For each weight:
  - loads graphs/routes/multimodal_<orig>_to_<dest>_<weight>.geojson (unless --route is provided),
  - saves EPS + PNG,
  - prints length summary by mode (and optionally saves CSVs).

Segments are colored by 'mode' (road/rail/connection). City markers and legend styles are configurable.

Examples
--------
# All weights, Hamburg→Munich
python 06_plot_multiweights_eps.py --orig Hamburg --dest Munich

# Only time + cost, labeled with Arial, more left padding, semi-opaque legend
python 06_plot_multiweights_eps.py --weights time,cost --legend-alpha 0.75 \
  --legend-font "Arial" --label-font "Arial" --extra-left 0.20

# Provide a specific route file (single weight)
python 06_plot_multiweights_eps.py --route graphs/routes/multimodal_hamburg_to_munich_time.geojson \
  --orig Hamburg --dest Munich --weights time
"""

from pathlib import Path
import argparse
import unicodedata
import warnings

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, LinearRing
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------- Paths & CRS ----------------
BASE_DIR = Path(".").resolve()
DEF_BORDER    = BASE_DIR / "input_files" / "germany_border_sta_2024.geojson"
DEF_ROUTESDIR = BASE_DIR / "graphs" / "routes"
DEF_PLOTSDIR  = BASE_DIR / "graphs" / "plots"

PLOT_EPSG = 25832   # meters (UTM32)
NET_EPSG  = 25832

# ---------------- STYLE DEFAULTS ----------------
DEFAULT_LABEL_FONTFAMILY  = "Arial"
DEFAULT_LABEL_FONTSIZE    = 18
DEFAULT_MARKER_SIZE       = 56

DEFAULT_LEGEND_FONTFAMILY = "Arial"
DEFAULT_LEGEND_FONTSIZE   = 18
DEFAULT_LEGEND_LOC        = "upper left"
DEFAULT_LEGEND_ALPHA      = 0.90

DEFAULT_EXTRA_LEFT_FRAC   = 0.1  # fraction of width to add left

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
    "nürnberg":   (11.0796, 49.4521)
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
    g = g[g.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    g = g.explode(index_parts=False).reset_index(drop=True)
    return g

def load_border(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    g = _ensure_crs(g)
    g = _to_plot_crs(g)
    return g

def route_default_path(orig: str, dest: str, weight: str, routes_dir: Path) -> Path:
    o = slugify(orig); d = slugify(dest); w = slugify(weight)
    return routes_dir / f"multimodal_{o}_to_{d}_{w}.geojson"

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

def to_plot_xy_from_lonlat(lon, lat):
    gs = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(PLOT_EPSG)
    p = gs.iloc[0]; return float(p.x), float(p.y)

def to_plot_xy_from_net(x, y):
    gs = gpd.GeoSeries([Point(x, y)], crs=NET_EPSG).to_crs(PLOT_EPSG)
    p = gs.iloc[0]; return float(p.x), float(p.y)

def resolve_marker_points(args):
    """Return ((ox,oy),(dx,dy), orig_label, dest_label) in PLOT_EPSG."""
    orig_label = args.orig
    dest_label = args.dest
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
    warnings.warn("City coordinates not resolved (unknown names and no --orig-xy/--dest-xy). Markers will be omitted.")
    return None, None, orig_label, dest_label

# ---------------- Length summaries ----------------
def _prep_segment_df(gdf: gpd.GeoDataFrame, dataset_label: str) -> pd.DataFrame:
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame(columns=["dataset", "mode", "length_m", "length_km"])
    if gdf.crs is None or gdf.crs.to_epsg() != PLOT_EPSG:
        gdf = gdf.to_crs(PLOT_EPSG)
    lengths_m = gdf.length.astype(float)
    modes = gdf["mode"].astype(str).str.lower() if "mode" in gdf.columns else pd.Series(["multimodal"]*len(gdf))
    df = pd.DataFrame({"dataset": dataset_label, "mode": modes.values, "length_m": lengths_m.values})
    df["length_km"] = df["length_m"] / 1000.0
    return df

def summarize_route_lengths(route_gdf, weight_label: str,
                            summary_csv: Path | None = None,
                            segments_csv: Path | None = None):
    seg = _prep_segment_df(route_gdf, f"multimodal_{weight_label}")
    if seg.empty:
        print(f"No features to summarize for {weight_label}.")
        return
    summary = (seg.groupby("mode", as_index=False)
                 .agg(segments=("length_m","size"), total_length_km=("length_km","sum")))
    total_km = seg["length_km"].sum()
    print(f"\n=== Length summary for {weight_label} ===")
    print(f"Total: {total_km:,.2f} km across {int(len(seg))} segments")
    for _, r in summary.sort_values("mode").iterrows():
        print(f"  {r['mode']:>11s}: {r['total_length_km']:,.2f} km  | segments: {int(r['segments'])}")
    print("=========================================\n")

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        out = summary.copy(); out.insert(0, "weight", weight_label)
        out.loc[len(out)] = [weight_label, "ALL", int(len(seg)), total_km]
        out.to_csv(summary_csv, index=False)
        print(f"🧾 Wrote summary CSV: {summary_csv}")
    if segments_csv is not None:
        segments_csv.parent.mkdir(parents=True, exist_ok=True)
        seg = seg.copy(); seg["segment_idx"] = range(len(seg))
        seg.insert(0, "weight", weight_label)
        seg.to_csv(segments_csv, index=False)
        print(f"🧾 Wrote per-segment CSV: {segments_csv}")

# ---------------- Plot (single route) ----------------
def plot_route_only(multi_gdf, border_gdf,
                    out_eps: Path, out_png: Path, dpi: int, title: str,
                    orig_marker, dest_marker, orig_label: str, dest_label: str,
                    label_fontfamily: str, label_fontsize: int, marker_size: int,
                    legend_fontfamily: str, legend_fontsize: int, legend_loc: str,
                    legend_alpha: float, extra_left_frac: float):
    fig, ax = plt.subplots(figsize=(9, 10))

    # Border
    if border_gdf is not None and len(border_gdf):
        try:
            border_gdf.plot(ax=ax, facecolor="none", edgecolor="dimgrey", linewidth=0.8, zorder=1)
        except Exception:
            border_gdf.boundary.plot(ax=ax, color="dimgrey", linewidth=0.8, zorder=1)

    # Multimodal by mode
    if multi_gdf is not None and len(multi_gdf):
        if "mode" in multi_gdf.columns:
            rr = multi_gdf[multi_gdf["mode"].astype(str).str.lower()=="road"]
            tt = multi_gdf[multi_gdf["mode"].astype(str).str.lower()=="rail"]
            cc = multi_gdf[multi_gdf["mode"].astype(str).str.lower()=="connection"]
            plotted_any = False
            if len(rr): rr.plot(ax=ax, linewidth=3, color="darkblue",      zorder=6, label="Road");        plotted_any = True
            if len(tt): tt.plot(ax=ax, linewidth=3, color="darkred",       zorder=6, label="Rail")
            if len(cc): cc.plot(ax=ax, linewidth=3, color="goldenrod", zorder=6, label="Connection")
            if not plotted_any and len(multi_gdf):
                multi_gdf.plot(ax=ax, linewidth=3, color="#16a34a", zorder=6, label="Route")
        else:
            multi_gdf.plot(ax=ax, linewidth=3, color="#16a34a", zorder=6, label="Route")

    # Extent (+ extra left padding)
    b_multi  = tuple(multi_gdf.total_bounds) if multi_gdf is not None and len(multi_gdf) else None
    b_border = tuple(border_gdf.total_bounds) if border_gdf is not None and len(border_gdf) else None
    extent   = compute_extent([b_border, b_multi], pad_ratio=0.06)
    if extent:
        width = extent[2] - extent[0]
        ax.set_xlim(extent[0] - extra_left_frac * width, extent[2])
        ax.set_ylim(extent[1], extent[3])

    # City markers & labels
    if orig_marker:
        ox, oy = orig_marker
        ax.scatter([ox],[oy], s=marker_size, color="#10b981", edgecolor="black", linewidth=0.5,
                   zorder=10, label="Origin")
        t = ax.annotate(orig_label, (ox, oy), xytext=(8,8), textcoords="offset points",
                        fontsize=label_fontsize, fontfamily=label_fontfamily, weight="bold", zorder=11)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])
    if dest_marker:
        dx, dy = dest_marker
        ax.scatter([dx],[dy], s=marker_size*1.1, color="#ef4444", edgecolor="black", linewidth=0.5,
                   marker="D", zorder=10, label="Destination")
        t = ax.annotate(dest_label, (dx, dy), xytext=(8,-10), textcoords="offset points",
                        fontsize=label_fontsize, fontfamily=label_fontfamily, weight="bold", zorder=11)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    ax.set_aspect("equal")
    # if title: ax.set_title(title)
    ax.axis("off")

    # Legend
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
    ap = argparse.ArgumentParser(description="Plot multimodal route(s) per weight (no road/rail comparison) with Germany border.")
    ap.add_argument("--orig", type=str, default="Berlin", help="Origin city label")
    ap.add_argument("--dest", type=str, default="Cottbus", help="Destination city label")
    ap.add_argument("--weights", type=str, default="distance,time,cost,emission",
                    help="Comma list of weights to plot. Defaults to all four.")
    ap.add_argument("--routes-dir", type=str, default=str(DEF_ROUTESDIR),
                    help="Where to look for default multimodal_<o>_to_<d>_<weight>.geojson")
    ap.add_argument("--route", type=str, default="",
                    help="Explicit single route file to plot (overrides default path)")

    ap.add_argument("--border", type=str, default=str(DEF_BORDER), help="Germany border GeoJSON")
    ap.add_argument("--out-dir", type=str, default=str(DEF_PLOTSDIR), help="Output directory for EPS/PNG")
    ap.add_argument("--dpi", type=int, default=200, help="PNG DPI")

    # City markers
    ap.add_argument("--orig-xy", type=float, nargs=2, help="Origin lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--dest-xy", type=float, nargs=2, help="Destination lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--crs", type=int, default=4326, choices=[4326, 25832], help="CRS for --orig-xy/--dest-xy")

    # Fonts / legend / padding
    ap.add_argument("--label-font", type=str, default=DEFAULT_LABEL_FONTFAMILY)
    ap.add_argument("--label-fontsize", type=int, default=DEFAULT_LABEL_FONTSIZE)
    ap.add_argument("--marker-size", type=int, default=DEFAULT_MARKER_SIZE)

    ap.add_argument("--legend-font", type=str, default=DEFAULT_LEGEND_FONTFAMILY)
    ap.add_argument("--legend-fontsize", type=int, default=DEFAULT_LEGEND_FONTSIZE)
    ap.add_argument("--legend-loc", type=str, default=DEFAULT_LEGEND_LOC)
    ap.add_argument("--legend-alpha", type=float, default=DEFAULT_LEGEND_ALPHA)
    ap.add_argument("--extra-left", type=float, default=DEFAULT_EXTRA_LEFT_FRAC,
                    help="Extra fraction of plot width to pad on the left")

    # CSV summaries
    ap.add_argument("--summary-csv", type=str, default="", help="Write per-weight length summaries to this CSV")
    ap.add_argument("--segments-csv", type=str, default="", help="Write per-segment lengths to this CSV")

    args = ap.parse_args()

    routes_dir = Path(args.routes_dir)
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    border_gdf = load_border(Path(args.border))

    # Resolve city markers (in plot CRS)
    o_xy, d_xy, o_label, d_label = resolve_marker_points(args)

    # Determine weights to iterate
    weights = [w.strip().lower() for w in args.weights.split(",") if w.strip()]
    if not weights:
        weights = ["distance"]

    # Optional CSV paths (append across weights)
    summary_csv  = Path(args.summary_csv)  if args.summary_csv  else None
    segments_csv = Path(args.segments_csv) if args.segments_csv else None
    if summary_csv and summary_csv.exists():  summary_csv.unlink()
    if segments_csv and segments_csv.exists(): segments_csv.unlink()

    for w in weights:
        # Route path
        if args.route:
            route_path = Path(args.route)   # explicit single route
        else:
            route_path = route_default_path(args.orig, args.dest, w, routes_dir)

        print(f"Loading route [{w}]: {route_path} ({'OK' if route_path.exists() else 'MISSING'})")
        route_gdf = load_route(route_path) if route_path.exists() else None
        if route_gdf is None or len(route_gdf) == 0:
            print(f"⚠️  Skipping weight '{w}': route not found or empty.")
            continue

        # Summaries (append)
        summarize_route_lengths(route_gdf, w,
                                summary_csv=(summary_csv if summary_csv else None),
                                segments_csv=(segments_csv if segments_csv else None))

        # Outputs per weight
        o_slug, d_slug = slugify(args.orig), slugify(args.dest)
        out_eps = out_dir / f"{o_slug}_to_{d_slug}_{w}.eps"
        out_png = out_dir / f"{o_slug}_to_{d_slug}_{w}.png"

        # Plot
        plot_route_only(
            route_gdf, border_gdf,
            out_eps, out_png, dpi=args.dpi, title=f"{args.orig} → {args.dest} [{w}]",
            orig_marker=o_xy, dest_marker=d_xy, orig_label=o_label, dest_label=d_label,
            label_fontfamily=args.label_font, label_fontsize=args.label_fontsize, marker_size=args.marker_size,
            legend_fontfamily=args.legend_font, legend_fontsize=args.legend_fontsize,
            legend_loc=args.legend_loc, legend_alpha=args.legend_alpha,
            extra_left_frac=args.extra_left
        )

    print("Done.")

if __name__ == "__main__":
    main()
