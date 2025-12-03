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
def compute_border_aspect_wh(border_path: Path, target_crs: str = "EPSG:3035") -> float:
    """Return width/height of Germany border bounds in target CRS."""
    if not border_path.exists():
        # Fallback: try roads+rails union
        raise FileNotFoundError(f"Border file not found: {border_path}")
    g = gpd.read_file(border_path)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(target_crs)
    minx, miny, maxx, maxy = g.total_bounds
    w = maxx - minx
    h = maxy - miny
    return float(w / h) if h > 0 else 1.0


def build_aspect_box_around_center(center_lon: float, center_lat: float,
                                   long_side_km: float, long_side_is: str,
                                   aspect_wh: float,
                                   target_crs: str = "EPSG:3035"):
    """
    Create a rectangle (Polygon) in target CRS centered at (lon,lat) whose
    width/height ratio equals `aspect_wh`.
    You choose which side is the 'long side' via `long_side_is` ('width' or 'height'),
    and its length in km; the other side is derived from the aspect.
    """
    # Center point in target CRS
    pt = gpd.GeoSeries([Point(center_lon, center_lat)], crs=4326).to_crs(target_crs).iloc[0]

    if long_side_is not in {"width", "height"}:
        long_side_is = "width"

    if long_side_is == "width":
        width_m  = long_side_km * 1000.0
        height_m = width_m / aspect_wh
    else:
        height_m = long_side_km * 1000.0
        width_m  = height_m * aspect_wh

    half_w = width_m  / 2.0
    half_h = height_m / 2.0
    return box(pt.x - half_w, pt.y - half_h, pt.x + half_w, pt.y + half_h)


def adjust_bbox_to_aspect(bbox_wgs84: tuple, aspect_wh: float,
                          target_crs: str = "EPSG:3035"):
    """
    Take an existing WGS84 bbox (min_lon,min_lat,max_lon,max_lat), convert to target CRS,
    and expand the SHORTER side about the bbox center so width/height == aspect_wh.
    Returns a Polygon in target CRS.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    poly = gpd.GeoSeries([box(min_lon, min_lat, max_lon, max_lat)], crs=4326).to_crs(target_crs).iloc[0]
    minx, miny, maxx, maxy = poly.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    w = maxx - minx
    h = maxy - miny
    if h == 0:
        h = 1e-6
    cur_aspect = w / h

    if abs(cur_aspect - aspect_wh) < 1e-6:
        return poly  # already matches

    if cur_aspect < aspect_wh:
        # too narrow -> widen to match
        new_w = aspect_wh * h
        half_w = new_w / 2.0
        return box(cx - half_w, cy - (h/2.0), cx + half_w, cy + (h/2.0))
    else:
        # too wide -> increase height to match
        new_h = w / aspect_wh
        half_h = new_h / 2.0
        return box(cx - (w/2.0), cy - half_h, cx + (w/2.0), cy + half_h)

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
    pois   = read_any_project(POIS_PATH)
    border = read_lines_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # If POIs aren’t points, plot centroids
    if not (pois.geom_type.isin(["Point", "MultiPoint"]).any()):
        pois = pois.copy()
        if not pois.empty:
            pois["geometry"] = pois.geometry.centroid

    # Simplify (lines only)
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

    # Clip to ROI (including POIs)
    roads, rails, conns, pois, border = clip_all(roi_poly, roads, rails, conns, pois, border)

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))
    ax.set_axis_off()

    # Draw order: roads → rail → POIs → connections → border (border on top, vector)
    # Draw order: roads → rail → POIs → connections → border (on top, vector)
    if not roads.empty:
        roads.plot(ax=ax, color=ROADS_COLOR, linewidth=ROADS_WIDTH, zorder=2,
                   rasterized=(RASTERIZE in ("roads", "all")))
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=3, rasterized=(RASTERIZE == "all"))
    if not pois.empty:
        pois.plot(ax=ax, markersize=POI_MS, color=POI_FACE, edgecolor=POI_EDGE, linewidth = POI_LW, zorder = 4, rasterized = (RASTERIZE == "all"))
    if not conns.empty:
        conns.plot(ax=ax, color=CONN_COLOR, linewidth=CONN_WIDTH, zorder=5,
                       rasterized=(RASTERIZE == "all"))
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=9, rasterized=False)
     # Zoom to ROI if used
    if roi_poly is not None:
        minx, miny, maxx, maxy = roi_poly.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # -------- Legend (upper-left), ~18pt --------
    handles = [
        Line2D([0], [0], color=ROADS_COLOR, lw=ROADS_WIDTH*2, label="Road"),
        Line2D([0], [0], color=RAIL_COLOR,  lw=RAIL_WIDTH*2,  label="Rail"),
        Line2D([0], [0], color=CONN_COLOR,  lw=CONN_WIDTH*2,  label="Connection"),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=POI_FACE, markeredgecolor=POI_EDGE,
               markersize=max(1, 15), label="POI"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    # Default: ~30 km around central Berlin
    # main(center="13.405,52.52", radius_km=15.0, no_clip=False)

    # Or: exact bbox
    # exact bbox instead of circle
    main(bbox="13.1834,52.3853,13.6266,52.6547", no_clip=False)


    # Or: no clipping
    # main(no_clip=True)
