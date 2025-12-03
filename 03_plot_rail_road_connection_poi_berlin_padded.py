#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box, Point

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR    = Path(__file__).resolve().parent
ROADS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS_PATH  = BASE_DIR / "graphs"     / "road_rail_connection_edges.geojson"
POIS_PATH   = BASE_DIR / "graphs"     / "pois_with_connections.gpkg"   # or input_files/final_pois.gpkg
BORDER_PATH = BASE_DIR / "input_files"/ "germany_border_sta_2024.geojson"

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_directconn_berlin_pois.png"
OUT_EPS = OUT_DIR / "roads_rail_directconn_berlin_pois.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# Styles
ROADS_COLOR = "#2c7fb8"; ROADS_WIDTH = 2
RAIL_COLOR  = "#e41a1c"; RAIL_WIDTH  = 2
CONN_COLOR  = "goldenrod"; CONN_WIDTH = 4
POI_FACE    = "green"; POI_EDGE = "black"; POI_MS = 64; POI_LW = 2
BORDER_COLOR= "black"; BORDER_WIDTH = 1.2

# --- SIZE CONTROLS ---
SIMPLIFY_M = 1
RASTERIZE  = None   # None | "roads" | "all"
PNG_DPI    = 300

# Fonts / path simplification
FONT_SIZE = 24
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "path.simplify": True,
    "path.simplify_threshold": 0.5,
})

# -----------------------------
# Helpers
# -----------------------------
def read_lines_project(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(TARGET_CRS)
    if (gdf.geom_type == "MultiLineString").any():
        gdf = gdf.explode(index_parts=False)
    return gdf

def read_any_project(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(TARGET_CRS)

def simplify_gdf(gdf: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    if gdf.empty or not tol_m or tol_m <= 0:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tol_m, preserve_topology=False)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

def bounds_union(*gdfs):
    tb = None
    for g in gdfs:
        if g is not None and not g.empty:
            if tb is None:
                tb = list(g.total_bounds)
            else:
                a, b, c, d = g.total_bounds
                tb[0] = min(tb[0], a); tb[1] = min(tb[1], b)
                tb[2] = max(tb[2], c); tb[3] = max(tb[3], d)
    return tuple(tb) if tb is not None else None

def compute_aspect_wh(bounds):
    minx, miny, maxx, maxy = bounds
    w = maxx - minx; h = maxy - miny
    return float(w / h) if h > 0 else 1.0

def pad_bounds_to_aspect(bounds, target_wh):
    """
    Pad bounds (minx,miny,maxx,maxy) to match target width/height ratio.
    Prefer vertical padding (top/bottom); if current is too tall, pad left/right.
    """
    minx, miny, maxx, maxy = bounds
    w = maxx - minx; h = maxy - miny
    if w <= 0 or h <= 0:
        return bounds
    cur_wh = w / h
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    if cur_wh > target_wh:
        # too wide -> increase height (white bars top/bottom)
        new_h = w / target_wh
        half_h = new_h / 2.0
        return (minx, cy - half_h, maxx, cy + half_h)
    elif cur_wh < target_wh:
        # too tall -> increase width (white bars left/right)
        new_w = h * target_wh
        half_w = new_w / 2.0
        return (cx - half_w, miny, cx + half_w, maxy)
    else:
        return bounds

def compute_germany_aspect(border_path: Path) -> float:
    g = gpd.read_file(border_path)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(TARGET_CRS)
    return compute_aspect_wh(g.total_bounds)

# -----------------------------
# Main
# -----------------------------
def main(
    bbox="",                         # optional WGS84 bbox string
    center="13.405,52.52",           # fallback center (Berlin)
    radius_km=30.0,                  # only used if bbox == ""
    no_clip=False
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    roads  = read_lines_project(ROADS_PATH)
    rails  = read_lines_project(RAILS_PATH)
    conns  = read_lines_project(CONNS_PATH)
    pois   = read_any_project(POIS_PATH)
    border = read_lines_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # POIs → points (centroid if needed)
    if not (pois.geom_type.isin(["Point", "MultiPoint"]).any()):
        pois = pois.copy()
        if not pois.empty:
            pois["geometry"] = pois.geometry.centroid

    # Simplify lines
    roads = simplify_gdf(roads, SIMPLIFY_M)
    rails = simplify_gdf(rails, SIMPLIFY_M * 0.8)
    conns = simplify_gdf(conns, SIMPLIFY_M * 0.6)

    # ROI (optional crop)
    if not no_clip:
        if bbox:
            parts = [float(x.strip()) for x in bbox.split(",")]
            assert len(parts) == 4, "--bbox must be 'min_lon,min_lat,max_lon,max_lat'"
            roi_poly = gpd.GeoSeries([box(*parts)], crs=4326).to_crs(TARGET_CRS).iloc[0]
        else:
            lon, lat = [float(x.strip()) for x in center.split(",")]
            pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(TARGET_CRS).iloc[0]
            roi_poly = pt.buffer(radius_km * 1000.0)
        roads, rails, conns, pois, border = clip_all(roi_poly, roads, rails, conns, pois, border)

    # --- Target aspect from Germany border (fallback: roads+rails) ---
    if BORDER_PATH.exists():
        target_wh = compute_germany_aspect(BORDER_PATH)
    else:
        uni = bounds_union(roads, rails)
        target_wh = compute_aspect_wh(uni) if uni else 1.0

    # --- Current content bounds (what you just produced) ---
    content_bounds = bounds_union(roads, rails, conns, pois)
    if content_bounds is None:
        raise RuntimeError("Nothing to plot after clipping; check ROI and inputs.")

    # --- Pad bounds to target aspect (adds white bars) ---
    view_bounds = pad_bounds_to_aspect(content_bounds, target_wh)

    # Figure with the SAME aspect as Germany
    fig_width = 8.0
    fig_height = fig_width / target_wh
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    ax.set_facecolor("white")  # white bars

    # Draw: roads → rail → POIs → connections → border (vector on top)
    if not roads.empty:
        roads.plot(ax=ax, color=ROADS_COLOR, linewidth=ROADS_WIDTH, zorder=2,
                   rasterized=(RASTERIZE in ("roads", "all")))
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=3,
                   rasterized=(RASTERIZE == "all"))
    if not pois.empty:
        pois.plot(ax=ax, markersize=POI_MS, color=POI_FACE, edgecolor=POI_EDGE,
                  linewidth=POI_LW, zorder=4, rasterized=(RASTERIZE == "all"))
    if not conns.empty:
        conns.plot(ax=ax, color=CONN_COLOR, linewidth=CONN_WIDTH, zorder=5,
                   rasterized=(RASTERIZE == "all"))
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=9, rasterized=False)

    # Center the current output and apply padded bounds
    minx, miny, maxx, maxy = view_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Legend (upper-left)
    handles = [
        Line2D([0], [0], color=ROADS_COLOR, lw=max(1.5, ROADS_WIDTH), label="Road"),
        Line2D([0], [0], color=RAIL_COLOR,  lw=max(1.5, RAIL_WIDTH),  label="Rail"),
        Line2D([0], [0], color=CONN_COLOR,  lw=max(1.5, CONN_WIDTH),  label="Connection"),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=POI_FACE, markeredgecolor=POI_EDGE,
               markersize=max(5, POI_MS * 0.25), label="POI"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    # Save (no tight bbox — keep white bars)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=PNG_DPI, facecolor="white")
    fig.savefig(OUT_EPS, format="eps", facecolor="white")
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

def clip_all(roi_poly, *gdfs):
    """Clip each GeoDataFrame to roi_poly (already in TARGET_CRS)."""
    clipped = []
    for g in gdfs:
        if g is None or g.empty or roi_poly is None:
            clipped.append(g)
        else:
            try:
                clipped.append(gpd.clip(g, roi_poly))
            except Exception:
                clipped.append(g)
    return clipped

if __name__ == "__main__":
    # Example: ~30km around central Berlin (or pass a bbox string)
    #main(center="13.405,52.52", radius_km=30.0, no_clip=False)
    main(bbox="13.1834,52.3853,13.6266,52.6547", no_clip=False)
