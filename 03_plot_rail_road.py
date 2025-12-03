#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR   = Path(__file__).resolve().parent
ROADS_PATH = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_PATH = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
BORDER_PATH= BASE_DIR / "input_files" / "germany_border_sta_2024.geojson"  # lines file (optional)

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_map.png"
OUT_EPS = OUT_DIR / "roads_rail_map.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# Styles
ROADS_COLOR = "#9e9e9e"
ROADS_WIDTH = 0.25
RAIL_COLOR  = "#2c7fb8"
RAIL_WIDTH  = 0.35
BORDER_COLOR= "#bdbdbd"
BORDER_WIDTH= 0.40

def read_and_project(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(TARGET_CRS)
    # Normalize multilines to lines (safer drawing)
    if (gdf.geom_type == "MultiLineString").any():
        gdf = gdf.explode(index_parts=False)
    return gdf

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roads  = read_and_project(ROADS_PATH)
    rails  = read_and_project(RAILS_PATH)
    border = read_and_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))  # width x (width*1.2)
    ax.set_axis_off()
    ax.set_title("Germany: Road + Rail Network", pad=10)

    # Draw (order matters): border → roads → rail
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=1)
    if not roads.empty:
        roads.plot(ax=ax, color=ROADS_COLOR, linewidth=ROADS_WIDTH, zorder=2)
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=3)

    # Legend
    handles = [
        Line2D([0], [0], color=ROADS_COLOR, lw=ROADS_WIDTH, label="Roads"),
        Line2D([0], [0], color=RAIL_COLOR,  lw=RAIL_WIDTH,  label="Rail"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)

    plt.tight_layout()

    # Save quick PNG + publication EPS
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    main()
