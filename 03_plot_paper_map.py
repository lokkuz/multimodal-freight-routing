#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR       = Path(__file__).resolve().parent
ROADS_PATH     = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_PATH     = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS_PATH     = BASE_DIR / "graphs"     / "road_rail_connection_edges.geojson"
POIS_CONN_PATH = BASE_DIR / "graphs"     / "pois_with_connections.gpkg"   # from the builder
BORDER_PATH    = BASE_DIR / "input_files"/ "germany_border_sta_2024.geojson"  # optional lines overlay

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_directconn_pois_connected.png"
OUT_EPS = OUT_DIR / "roads_rail_directconn_pois_connected.eps"

TARGET_CRS = "EPSG:3035"  # Europe LAEA (meters)

# --- Display-only label for your thresholds (doesn't affect data) ---
RAIL_THRESH_KM = 3.0
ROAD_THRESH_KM = 10.0

# Styles
ROADS_COLOR = "#9e9e9e"; ROADS_WIDTH = 0.25
RAIL_COLOR  = "#2c7fb8"; RAIL_WIDTH  = 0.35
CONN_COLOR  = "#d95f0e"; CONN_WIDTH  = 0.60
POI_FACE    = "#111111"; POI_EDGE    = "white"; POI_MS = 6; POI_LW = 0.3
BORDER_COLOR= "#bdbdbd"; BORDER_WIDTH= 0.40

def read_and_project_lines(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(TARGET_CRS)
    if (gdf.geom_type == "MultiLineString").any():
        gdf = gdf.explode(index_parts=False)
    return gdf

def read_and_project_any(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(TARGET_CRS)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roads  = read_and_project_lines(ROADS_PATH)
    rails  = read_and_project_lines(RAILS_PATH)
    conns  = read_and_project_lines(CONNS_PATH)
    pois_c = read_and_project_any(POIS_CONN_PATH)
    border = read_and_project_lines(BORDER_PATH) if BORDER_PATH.exists() else None

    # If POIs with connections aren’t points (unlikely), plot centroids
    if not (pois_c.geom_type.isin(["Point", "MultiPoint"]).any()):
        pois_c = pois_c.copy()
        pois_c["geometry"] = pois_c.geometry.centroid

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))
    ax.set_axis_off()
    ax.set_title(f"Germany: Roads, Rail, Direct Connections & POIs (Rail ≤ {RAIL_THRESH_KM:.0f} km, Road ≤ {ROAD_THRESH_KM:.0f} km)",
                 pad=10)

    # Draw in order: border → roads → rail → connections → POIs(with connections)
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=1)
    if not roads.empty:
        roads.plot(ax=ax, color=ROADS_COLOR, linewidth=ROADS_WIDTH, zorder=2)
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=3)
    if not conns.empty:
        conns.plot(ax=ax, color=CONN_COLOR, linewidth=CONN_WIDTH, zorder=4)
    if not pois_c.empty:
        pois_c.plot(ax=ax, markersize=POI_MS, color=POI_FACE, edgecolor=POI_EDGE,
                    linewidth=POI_LW, zorder=5)

    # Legend (upper left)
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
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    main()
