#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_plot_routes_eps.py

Plot (in one figure) the road-only route, rail-only route, your multimodal route,
and the Germany border. Saves BOTH a vector EPS and a PNG preview.

Also prints a length summary and (optionally) writes:
- an aggregate CSV (--summary-csv) with total length & segment count per dataset/mode
- a per-segment CSV (--segments-csv) with each section's length

New:
- Style controls for legend and city labels (font family/size, legend location)
  configurable at the top or via CLI flags.
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
DEF_BORDER = BASE_DIR / "input_files" / "germany_border_sta_2024.geojson"
DEF_ROUTES_DIR = BASE_DIR / "graphs" / "routes"
DEF_PLOTS_DIR = BASE_DIR / "graphs" / "plots"

PLOT_EPSG = 25832  # UTM32N (meters)
NET_EPSG  = 25832  # for --crs 25832 inputs

# ---------------- STYLE DEFAULTS (edit here if you like) ----------------
DEFAULT_LABEL_FONTFAMILY  = "Arial"
DEFAULT_LABEL_FONTSIZE    = 18
DEFAULT_MARKER_SIZE       = 56

DEFAULT_LEGEND_FONTFAMILY = "Arial"
DEFAULT_LEGEND_FONTSIZE   = 18
DEFAULT_LEGEND_LOC        = "upper left"   # e.g. 'upper left','upper right','lower right','upper center','center'

# ---------------- Gazetteer (lon, lat WGS84) ----------------
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

def default_route_paths(orig: str, dest: str, metric: str, routes_dir: Path) -> tuple[Path, Path, Path]:
    o = slugify(orig); d = slugify(dest); m = slugify(metric)
    road = routes_dir / f"road_{o}_to_{d}_{m}.geojson"
    rail = routes_dir / f"rail_{o}_to_{d}_{m}.geojson"
    multi = routes_dir / f"multimodal_{o}_to_{d}.geojson"  # optional convenience
    return road, rail, multi

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
        pad = 10000.0  # 10 km if degenerate
        return (minx - pad, miny - pad, maxx + pad, maxy + pad)
    px = dx * pad_ratio; py = dy * pad_ratio
    return (minx - px, miny - py, maxx + px, maxy + py)

def to_plot_xy_from_lonlat(lon, lat):
    gs = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(PLOT_EPSG)
    p = gs.iloc[0]
    return float(p.x), float(p.y)

def to_plot_xy_from_net(x, y):
    gs = gpd.GeoSeries([Point(x, y)], crs=NET_EPSG).to_crs(PLOT_EPSG)
    p = gs.iloc[0]
    return float(p.x), float(p.y)

def resolve_marker_points(args):
    """Return ((ox,oy),(dx,dy), orig_label, dest_label) in PLOT_EPSG."""
    orig_label = args.orig
    dest_label = args.dest #"Frankfurt am Main"#

    if args.orig_xy and args.dest_xy:
        ox, oy = float(args.orig_xy[0]), float(args.orig_xy[1])
        dx, dy = float(args.dest_xy[0]), float(args.dest_xy[1])
        if args.crs == 4326:
            o_xy = to_plot_xy_from_lonlat(ox, oy)
            d_xy = to_plot_xy_from_lonlat(dx, dy)
        else:  # 25832
            o_xy = to_plot_xy_from_net(ox, oy)
            d_xy = to_plot_xy_from_net(dx, dy)
        return o_xy, d_xy, orig_label, dest_label

    lo = CITY_WGS84.get(args.orig.strip().lower())
    ld = CITY_WGS84.get(args.dest.strip().lower())
    if lo and ld:
        o_xy = to_plot_xy_from_lonlat(lo[0], lo[1])
        d_xy = to_plot_xy_from_lonlat(ld[0], ld[1])
        return o_xy, d_xy, orig_label, dest_label

    warnings.warn("City coordinates not resolved (unknown names and no --orig-xy/--dest-xy). Markers will be omitted.")
    return None, None, orig_label, dest_label

# ---------------- Length summaries ----------------
def _prep_segment_df(gdf: gpd.GeoDataFrame, dataset_label: str) -> pd.DataFrame:
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame(columns=["dataset", "mode", "length_m", "length_km"])
    if gdf.crs is None or gdf.crs.to_epsg() != PLOT_EPSG:
        gdf = gdf.to_crs(PLOT_EPSG)
    lengths_m = gdf.length.astype(float)
    if "mode" in gdf.columns:
        modes = gdf["mode"].astype(str).str.lower()
    else:
        modes = pd.Series([dataset_label] * len(gdf), index=gdf.index)
    df = pd.DataFrame({
        "dataset": dataset_label,
        "mode": modes.values,
        "length_m": lengths_m.values
    })
    df["length_km"] = df["length_m"] / 1000.0
    return df

def summarize_lengths(road_gdf, rail_gdf, multi_gdf,
                      summary_csv: Path | None = None,
                      segments_csv: Path | None = None):
    seg_road  = _prep_segment_df(road_gdf,  "road")
    seg_rail  = _prep_segment_df(rail_gdf,  "rail")
    seg_multi = _prep_segment_df(multi_gdf, "multimodal")

    seg_all = pd.concat([seg_road, seg_rail, seg_multi], ignore_index=True)
    if seg_all.empty:
        print("No route layers to summarize.")
        return

    summary = (seg_all
               .groupby(["dataset", "mode"], as_index=False)
               .agg(segments=("length_m", "size"),
                    total_length_km=("length_km", "sum")))

    totals = (summary
              .groupby("dataset", as_index=False)
              .agg(segments=("segments", "sum"),
                   total_length_km=("total_length_km", "sum")))

    def fmt(km): return f"{km:,.2f} km"
    print("\n=== Length summary (by dataset & mode) ===")
    if not summary.empty:
        for ds in ["road", "rail", "multimodal"]:
            sds = summary[summary["dataset"] == ds]
            if sds.empty:
                continue
            total_km = totals.loc[totals["dataset"]==ds, "total_length_km"].values[0]
            total_segs = int(totals.loc[totals["dataset"]==ds, "segments"].values[0])
            print(f"{ds.upper():-^42}")
            print(f"Total: {fmt(total_km)} across {total_segs} segments")
            for _, r in sds.sort_values("mode").iterrows():
                print(f"  {r['mode']:>12s}: {fmt(r['total_length_km'])}  | segments: {int(r['segments'])}")
    print("=========================================\n")

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        out = summary.copy()
        out_tot = totals.copy(); out_tot["mode"] = "ALL"
        out = pd.concat([out, out_tot], ignore_index=True)
        out.to_csv(summary_csv, index=False)
        print(f"🧾 Wrote summary CSV: {summary_csv}")

    if segments_csv is not None:
        segments_csv.parent.mkdir(parents=True, exist_ok=True)
        seg_all = seg_all.copy()
        seg_all["segment_idx"] = seg_all.groupby("dataset").cumcount()
        cols = ["dataset", "segment_idx", "mode", "length_m", "length_km"]
        seg_all[cols].to_csv(segments_csv, index=False)
        print(f"🧾 Wrote per-segment CSV: {segments_csv}")

# ---------------- Plot ----------------
def plot_all(road_gdf, rail_gdf, multi_gdf, border_gdf,
             out_eps: Path, out_png: Path, dpi: int, title: str,
             orig_marker, dest_marker, orig_label: str, dest_label: str,
             label_fontfamily: str, label_fontsize: int, marker_size: int,
             legend_fontfamily: str, legend_fontsize: int, legend_loc: str):
    fig, ax = plt.subplots(figsize=(9, 10))

    # 1) Germany border
    if border_gdf is not None and len(border_gdf):
        try:
            border_gdf.plot(ax=ax, facecolor="none", edgecolor="dimgrey", linewidth=0.8, zorder=1)
        except Exception:
            border_gdf.boundary.plot(ax=ax, color="dimgrey", linewidth=0.8, zorder=1)

    # 2) Road & Rail routes
    if road_gdf is not None and len(road_gdf):
        road_gdf.plot(ax=ax, linewidth=0.8, color="darkblue", zorder=7, label="Road route")
    if rail_gdf is not None and len(rail_gdf):
        rail_gdf.plot(ax=ax, linewidth=0.8, color="darkred", zorder=7, label="Rail route")

    # 3) Multimodal route (by mode if available)
    if multi_gdf is not None and len(multi_gdf):
        if "mode" in multi_gdf.columns:
            mm = multi_gdf
            rr = mm[mm["mode"].astype(str).str.lower()=="road"]
            tt = mm[mm["mode"].astype(str).str.lower()=="rail"]
            cc = mm[mm["mode"].astype(str).str.lower()=="connection"]
            plotted_any = False
            if len(rr): rr.plot(ax=ax, linewidth=4, color="blue", zorder=6, label="Multimodal road"); plotted_any = True
            if len(tt): tt.plot(ax=ax, linewidth=4, color="red", zorder=6, label="Multimodal rail")
            if len(cc): cc.plot(ax=ax, linewidth=4, color="goldenrod", zorder=6, label="Connection")
            if not plotted_any and len(mm):
                mm.plot(ax=ax, linewidth=2.8, color="#16a34a", zorder=6, label="Multimodal route")
        else:
            multi_gdf.plot(ax=ax, linewidth=2.8, color="#16a34a", zorder=6, label="Multimodal route")

    # Extent
    b_road   = tuple(road_gdf.total_bounds)  if road_gdf  is not None and len(road_gdf)  else None
    b_rail   = tuple(rail_gdf.total_bounds)  if rail_gdf  is not None and len(rail_gdf)  else None
    b_multi  = tuple(multi_gdf.total_bounds) if multi_gdf is not None and len(multi_gdf) else None
    b_border = tuple(border_gdf.total_bounds) if border_gdf is not None and len(border_gdf) else None
    extent = compute_extent([b_border, b_road, b_rail, b_multi], pad_ratio=0.06)
    if extent:
        extra_left = 0.15 * (extent[2] - extent[0])  # 10% of width, tweak as needed
        ax.set_xlim(extent[0] - extra_left, extent[2])
        ax.set_ylim(extent[1], extent[3])

    # 4) City markers & labels
    if orig_marker:
        ox, oy = orig_marker
        ax.scatter([ox], [oy], s=marker_size, color="#10b981", edgecolor="black", linewidth=0.5,
                   zorder=10, label="Origin")
        t = ax.annotate(orig_label, (ox, oy), xytext=(8, 8), textcoords="offset points",
                        fontsize=label_fontsize, fontfamily=label_fontfamily,
                        weight="bold", zorder=11)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])
    if dest_marker:
        dx, dy = dest_marker
        ax.scatter([dx], [dy], s=marker_size*1.1, color="#ef4444", edgecolor="black", linewidth=0.5,
                   marker="D", zorder=10, label="Destination")
        t = ax.annotate(dest_label, (dx, dy), xytext=(8, -10), textcoords="offset points",
                        fontsize=label_fontsize, fontfamily=label_fontfamily,
                        weight="bold", zorder=11)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])

    ax.set_aspect("equal")
    # if title:
    #     ax.set_title(title)
    ax.axis("off")

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        leg = ax.legend(loc=legend_loc,
                        frameon=True,
                        facecolor="white",
                        framealpha=0.9,  # 0=fully transparent, 1=opaque
                        prop={"family": legend_fontfamily, "size": legend_fontsize})

    out_eps.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_eps, format="eps", bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ EPS saved:  {out_eps}")
    print(f"✅ PNG saved:  {out_png}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot road, rail, multimodal routes with Germany border into EPS+PNG, with city labels and length summaries.")
    ap.add_argument("--orig", type=str, default="Hamburg", help="Origin city label (used for default files/title/labels)")
    ap.add_argument("--dest", type=str, default="Munich", help="Destination city label")
    ap.add_argument("--metric", type=str, default="distance", help="Used in default route filenames (distance|time)")

    # Optional explicit files (overrides defaults)
    ap.add_argument("--road", type=str, default="", help="Path to road route GeoJSON")
    ap.add_argument("--rail", type=str, default="", help="Path to rail route GeoJSON")
    ap.add_argument("--multimodal", type=str, default="", help="Path to multimodal route (GeoJSON/GPKG). Optional.")
    ap.add_argument("--border", type=str, default=str(DEF_BORDER), help="Germany border GeoJSON")

    # Outputs
    ap.add_argument("--out", type=str, default="", help="Output EPS path (PNG saved next to it unless --out-png)")
    ap.add_argument("--out-png", type=str, default="", help="Optional explicit PNG output path")
    ap.add_argument("--dpi", type=int, default=200, help="PNG DPI")

    # City marker inputs
    ap.add_argument("--orig-xy", type=float, nargs=2, help="Origin coords: lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--dest-xy", type=float, nargs=2, help="Destination coords: lon lat (EPSG:4326) or x y with --crs 25832")
    ap.add_argument("--crs", type=int, default=4326, choices=[4326, 25832], help="CRS for --orig-xy/--dest-xy")
    ap.add_argument("--label-font", type=str, default=DEFAULT_LABEL_FONTFAMILY, help="Font family for city labels")
    ap.add_argument("--label-fontsize", type=int, default=DEFAULT_LABEL_FONTSIZE, help="City label font size")
    ap.add_argument("--marker-size", type=int, default=DEFAULT_MARKER_SIZE, help="City marker size")

    # NEW: legend style controls
    ap.add_argument("--legend-font", type=str, default=DEFAULT_LEGEND_FONTFAMILY, help="Legend font family")
    ap.add_argument("--legend-fontsize", type=int, default=DEFAULT_LEGEND_FONTSIZE, help="Legend font size")
    ap.add_argument("--legend-loc", type=str, default=DEFAULT_LEGEND_LOC,
                    help="Legend location (e.g. 'upper left','upper right','lower left','lower right','upper center','center')")

    # CSV outputs
    ap.add_argument("--summary-csv", type=str, default="", help="Optional CSV path for aggregate length summary")
    ap.add_argument("--segments-csv", type=str, default="", help="Optional CSV path for per-segment lengths")

    args = ap.parse_args()

    # Resolve route files if explicit not given
    road_path = Path(args.road) if args.road else None
    rail_path = Path(args.rail) if args.rail else None
    multi_path = Path(args.multimodal) if args.multimodal else None

    if road_path is None or rail_path is None:
        d_road, d_rail, d_multi = default_route_paths(args.orig, args.dest, args.metric, DEF_ROUTES_DIR)
        road_path = road_path or d_road
        rail_path = rail_path or d_rail
        multi_path = multi_path or d_multi

    # Output paths
    if args.out:
        out_eps = Path(args.out)
        if out_eps.suffix.lower() != ".eps":
            out_eps = out_eps.with_suffix(".eps")
    else:
        out_eps = DEF_PLOTS_DIR / f"{slugify(args.orig)}_to_{slugify(args.dest)}_{slugify(args.metric)}.eps"

    if args.out_png:
        out_png = Path(args.out_png)
        if out_png.suffix.lower() != ".png":
            out_png = out_png.with_suffix(".png")
    else:
        out_png = out_eps.with_suffix(".png")

    # Load layers
    print(f"Loading border: {args.border}")
    border_gdf = load_border(Path(args.border))

    print(f"Loading road route: {road_path} ({'OK' if road_path.exists() else 'MISSING'})")
    road_gdf = load_route(road_path) if road_path.exists() else None

    print(f"Loading rail route: {rail_path} ({'OK' if rail_path.exists() else 'MISSING'})")
    rail_gdf = load_route(rail_path) if rail_path.exists() else None

    print(f"Loading multimodal route: {multi_path} ({'OK' if multi_path.exists() else 'MISSING'})")
    multi_gdf = None
    if multi_path.exists():
        if multi_path.suffix.lower() in [".gpkg", ".geopackage"]:
            try:
                mm = gpd.read_file(multi_path, layer="network_edges")
            except Exception:
                mm = gpd.read_file(multi_path)
        else:
            mm = gpd.read_file(multi_path)
        mm = _ensure_crs(mm)
        mm = _explode_lines_keep_lines(mm)
        mm = _to_plot_crs(mm)
        multi_gdf = mm[mm.geometry.geom_type.isin(["LineString", "MultiLineString"])].explode(index_parts=False).reset_index(drop=True)

    # Resolve city markers (in PLOT_EPSG)
    o_xy, d_xy, o_label, d_label = resolve_marker_points(args)

    # Title
    title = f"{args.orig} \u2192 {args.dest} ({args.metric})"

    # Summaries
    summary_csv = Path(args.summary_csv) if args.summary_csv else None
    segments_csv = Path(args.segments_csv) if args.segments_csv else None
    summarize_lengths(road_gdf, rail_gdf, multi_gdf, summary_csv=summary_csv, segments_csv=segments_csv)

    # Plot & save
    plot_all(road_gdf, rail_gdf, multi_gdf, border_gdf,
             out_eps, out_png, dpi=args.dpi, title=title,
             orig_marker=o_xy, dest_marker=d_xy,
             orig_label=o_label, dest_label=d_label,
             label_fontfamily=args.label_font, label_fontsize=args.label_fontsize, marker_size=args.marker_size,
             legend_fontfamily=args.legend_font, legend_fontsize=args.legend_fontsize, legend_loc=args.legend_loc)

if __name__ == "__main__":
    main()
