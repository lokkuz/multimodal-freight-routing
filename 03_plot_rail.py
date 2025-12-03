#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
RAILS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
BORDER_PATH = BASE_DIR / "input_files" / "germany_border_sta_2024.geojson"  # lines file
OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "rail_only_map.png"
OUT_EPS = OUT_DIR / "rail_only_map.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# Simple styles
RAIL_COLOR   = "#2c7fb8"
RAIL_WIDTH   = 0.30
BORDER_COLOR = "#9e9e9e"
BORDER_WIDTH = 0.40

def read_and_project(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        # assume WGS84 if missing
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(TARGET_CRS)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load & project
    rails  = read_and_project(RAILS_PATH)
    # Border is optional—only for visual context (no clipping)
    border = read_and_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))  # width x (width*1.2)
    ax.set_axis_off()
    ax.set_title("Germany Rail Network", pad=10)

    # Draw (border below rails)
    if border is not None and not border.empty:
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=1)
    if not rails.empty:
        rails.plot(ax=ax, color=RAIL_COLOR, linewidth=RAIL_WIDTH, zorder=2)

    plt.tight_layout()

    # Save quick PNG + publication EPS
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    main()
