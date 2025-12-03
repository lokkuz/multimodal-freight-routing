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
POIS_PATH   = BASE_DIR / "graphs" / "pois_with_connections.gpkg"                   # points (or polygons)
BORDER_PATH = BASE_DIR / "input_files" / "germany_border_sta_2024.geojson"   # optional lines overlay

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_pois_small.png"
OUT_EPS = OUT_DIR / "roads_rail_pois_small.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# -----------------------------
# SIZE CONTROLS (tune these)
# -----------------------------
SIMPLIFY_M = 1         # simplify geometry in meters (try 75–150). Higher -> smaller EPS.
RASTERIZE  = "roads"     # None | "roads" | "all"  (bitmap-embed heavy layers in EPS)
PNG_DPI    = 300

# Global font size (~20 pt) and extra path simplification
FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,           # all text (incl. legend)
    "legend.fontsize": FONT_SIZE,     # explicit legend size
    "font.family": "sans-serif",      # or "serif"
    "font.sans-serif": ["Arial"],  # or ["Arial"], ["Helvetica"], ...
   # "font.family": "serif",      # or "serif"
   # 'font.serif': ['Times New Roman'],  # if you use serif
    "path.simplify": True,
    "path.simplify_threshold": 0.5,   # 0.5 = good size reduction; 0.0 disables it
})

# Styles
ROADS_COLOR = "#2c7fb8"; ROADS_WIDTH = 0.2 #" #2c7fb8"
RAIL_COLOR  = "#e41a1c"; RAIL_WIDTH  = 0.5
POI_FACE    = "green"; POI_EDGE    = "black"; POI_MS = 14; POI_LW = 0.6   # small dots
BORDER_COLOR= "black"; BORDER_WIDTH= 1.2

def read_and_project_lines(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(TARGET_CRS)
    if (gdf.geom_type == "MultiLineString").any():
        gdf = gdf.explode(index_parts=False)
    return gdf

def read_and_project_any(path: Path) -> gpd.GeoDataFrame:
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

    roads  = read_and_project_lines(ROADS_PATH)
    rails  = read_and_project_lines(RAILS_PATH)
    pois   = read_and_project_any(POIS_PATH)
    border = read_and_project_lines(BORDER_PATH) if BORDER_PATH.exists() else None

    # If POIs aren’t points (e.g., polygons), plot their centroids
    if not (pois.geom_type.isin(["Point", "MultiPoint"]).any()):
        pois = pois.copy()
        if not pois.empty:
            pois["geometry"] = pois.geometry.centroid

    # Simplify linework to keep EPS small
    roads = simplify_gdf(roads, SIMPLIFY_M)
    rails = simplify_gdf(rails, max(1, int(SIMPLIFY_M * 0.8)))  # keep rail a touch crisper

    # Figure (no title)
    fig, ax = plt.subplots(figsize=(8.0, 9.6))  # width x (width*1.2)
    ax.set_axis_off()

    # Draw in order:  roads → rail → POIs → border
    if not roads.empty:
        roads.plot(ax=ax, color=ROADS_COLOR, linewidth=ROADS_WIDTH, zorder=2,
                   rasterized=(RASTERIZE in ("roads", "all")))
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=3,
                   rasterized=(RASTERIZE == "all"))
    if not pois.empty:
        pois.plot(ax=ax, markersize=POI_MS, color=POI_FACE, edgecolor=POI_EDGE,
                  linewidth=POI_LW, zorder=4, rasterized=(RASTERIZE == "all"))
    if border is not None and not border.empty:
        # Optional: light simplify for border too
        border = simplify_gdf(border, max(1, int(SIMPLIFY_M * 0.8)))
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=9,
                    rasterized=(RASTERIZE == "all"))
    # Legend (upper-left), ~20pt
    handles = [
        Line2D([0], [0], color=ROADS_COLOR, lw=RAIL_WIDTH* 4, label="Road"),
        Line2D([0], [0], color=RAIL_COLOR,  lw=RAIL_WIDTH* 4,  label="Rail"),
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=POI_FACE, markeredgecolor=POI_EDGE,
               markersize=POI_MS * 0.8, label="POI"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, prop={"size": FONT_SIZE})

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    main()
