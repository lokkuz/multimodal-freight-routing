#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR    = Path(__file__).resolve().parent
ROADS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS_PATH  = BASE_DIR / "graphs"     / "road_rail_connection_edges.geojson"
POIS_PATH   = BASE_DIR / "graphs"     / "pois_with_connections.gpkg"  # only connected POIs
BORDER_PATH = BASE_DIR / "input_files"/ "germany_border_sta_2024.geojson"  # optional lines overlay

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_directconn_pois_conn_on_top.png"
OUT_EPS = OUT_DIR / "roads_rail_directconn_pois_conn_on_top.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# Styles
ROADS_COLOR = "#9e9e9e"; ROADS_WIDTH = 0.25
RAIL_COLOR  = "#2c7fb8"; RAIL_WIDTH  = 0.35
CONN_COLOR  = "#d95f0e"; CONN_WIDTH  = 0.60
POI_FACE    = "#111111"; POI_EDGE    = "white"; POI_MS = 3; POI_LW = 0.25   # smaller dots
BORDER_COLOR= "#bdbdbd"; BORDER_WIDTH= 0.40

# --- SIZE CONTROLS ---
SIMPLIFY_M = 75        # meters; raise (e.g., 100–150) to shrink EPS more
RASTERIZE  = "roads"   # None | "roads" | "all"
PNG_DPI    = 300

# Matplotlib path simplification (helps vector size)
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 0.5

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

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roads  = read_lines_project(ROADS_PATH)
    rails  = read_lines_project(RAILS_PATH)
    conns  = read_lines_project(CONNS_PATH)
    pois   = read_any_project(POIS_PATH)  # connected POIs only
    border = read_lines_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # If POIs aren’t points (rare), plot centroids
    if not (pois.geom_type.isin(["Point", "MultiPoint"]).any()):
        pois = pois.copy()
        if not pois.empty:
            pois["geometry"] = pois.geometry.centroid

    # Simplify linework to keep EPS small
    roads = simplify_gdf(roads, SIMPLIFY_M)
    rails = simplify_gdf(rails, SIMPLIFY_M * 0.8)
    conns = simplify_gdf(conns, SIMPLIFY_M * 0.6)

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))
    ax.set_axis_off()
    ax.set_title("Germany: Roads, Rail, Direct Connections & POIs", pad=10)

    # Draw order: border → roads → rail → POIs → connections (connections OVER POIs)
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=1)
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

    # Legend (upper-left)
    handles = [
        Line2D([0], [0], color=ROADS_COLOR, lw=ROADS_WIDTH, label="Roads"),
        Line2D([0], [0], color=RAIL_COLOR,  lw=RAIL_WIDTH,  label="Rail"),
        Line2D([0], [0], color=CONN_COLOR,  lw=CONN_WIDTH,  label="Direct connections"),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=POI_FACE, markeredgecolor=POI_EDGE,
               markersize=POI_MS * 0.6, label="POIs (connected)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    main()
