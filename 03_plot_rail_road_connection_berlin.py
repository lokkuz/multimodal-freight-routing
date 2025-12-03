#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR    = Path(__file__).resolve().parent
ROADS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS_PATH  = BASE_DIR / "graphs"     / "road_rail_connection_edges.geojson"
BORDER_PATH = BASE_DIR / "input_files"/ "germany_border_sta_2024.geojson"  # optional lines overlay

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_directconn_berlin.png"
OUT_EPS = OUT_DIR / "roads_rail_directconn_berlin.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# Styles
ROADS_COLOR = "#2c7fb8"; ROADS_WIDTH = 0.2
RAIL_COLOR  = "#e41a1c"; RAIL_WIDTH  = 0.5
CONN_COLOR  = "goldenrod"; CONN_WIDTH = 1.2
BORDER_COLOR= "black"; BORDER_WIDTH   = 1.2

# --- SIZE CONTROLS ---
SIMPLIFY_M = 1
RASTERIZE  = "roads"   # None | "roads" | "all"
PNG_DPI    = 300

# Fonts / path simplification
FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "path.simplify": True,
    "path.simplify_threshold": 0.5,
})

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

def simplify_gdf(gdf: gpd.GeoDataFrame, tol_m: float) -> gpd.GeoDataFrame:
    if gdf.empty or not tol_m or tol_m <= 0:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tol_m, preserve_topology=False)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf

def build_roi_polygon_3035(bbox_wgs84=None, center_wgs84=None, radius_km=None):
    """Return a Polygon in TARGET_CRS to clip with."""
    if bbox_wgs84 is not None:
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        roi = gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs=4326).to_crs(TARGET_CRS)
        return roi.geometry.iloc[0]
    if center_wgs84 is not None and radius_km:
        lon, lat = center_wgs84
        pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(TARGET_CRS).iloc[0]
        return pt.buffer(radius_km * 1000.0)
    return None

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

def main(
    bbox="",
    center="13.405,52.52",  # Berlin Mitte approx
    radius_km=30.0,
    no_clip=False
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    roads  = read_lines_project(ROADS_PATH)
    rails  = read_lines_project(RAILS_PATH)
    conns  = read_lines_project(CONNS_PATH)
    border = read_lines_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # Simplify
    roads = simplify_gdf(roads, SIMPLIFY_M)
    rails = simplify_gdf(rails, SIMPLIFY_M * 0.8)
    conns = simplify_gdf(conns, SIMPLIFY_M * 0.6)

    # Build ROI
    bbox_vals = None
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        assert len(parts) == 4, "--bbox must be 'min_lon,min_lat,max_lon,max_lat'"
        bbox_vals = parts
    lon, lat = [float(x.strip()) for x in center.split(",")]
    roi_poly = None if no_clip else build_roi_polygon_3035(bbox_vals, (lon, lat), radius_km if not bbox_vals else None)

    # Clip
    roads, rails, conns, border = clip_all(roi_poly, roads, rails, conns, border)

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))
    ax.set_axis_off()

    # Draw order: roads → rail → connections → border (on top, vector)
    if not roads.empty:
        roads.plot(ax=ax, color=ROADS_COLOR, linewidth=ROADS_WIDTH, zorder=2,
                   rasterized=(RASTERIZE in ("roads", "all")))
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=3,
                   rasterized=(RASTERIZE == "all"))
    if not conns.empty:
        conns.plot(ax=ax, color=CONN_COLOR, linewidth=CONN_WIDTH, zorder=4,
                   rasterized=(RASTERIZE == "all"))
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=9, rasterized=False)

    # Focus the view tightly on ROI bounds (if used)
    if roi_poly is not None:
        minx, miny, maxx, maxy = roi_poly.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    # --- adjust these defaults as needed ---
    # Example 1 (default): center/radius ~ Berlin
    main(center="13.405,52.52", radius_km=30.0, no_clip=False)

    # Example 2: exact bbox instead of circle
    # main(bbox="13.0,52.3,13.9,52.7", no_clip=False)

    # Example 3: disable clipping
    # main(no_clip=True)
