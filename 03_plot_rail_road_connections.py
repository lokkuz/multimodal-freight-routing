#!/usr/bin/env python3
# pip install geopandas shapely matplotlib

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# -----------------------------
# EDIT THESE PATHS IF NEEDED
# -----------------------------
BASE_DIR    = Path(__file__).resolve().parent
ROADS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS_PATH  = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS_PATH  = BASE_DIR / "graphs"     / "road_rail_connection_edges.geojson"
BORDER_PATH = BASE_DIR / "input_files"/ "germany_border_sta_2024.geojson"  # optional lines overlay

OUT_DIR = BASE_DIR / "figures"
OUT_PNG = OUT_DIR / "roads_rail_directconn.png"
OUT_EPS = OUT_DIR / "roads_rail_directconn.eps"

# Map projection for Europe (meters)
TARGET_CRS = "EPSG:3035"

# Styles
ROADS_COLOR = "#2c7fb8"; ROADS_WIDTH = 0.2 #" #2c7fb8"
RAIL_COLOR  = "#e41a1c"; RAIL_WIDTH  = 0.5
CONN_COLOR  = "goldenrod"; CONN_WIDTH  = 1.2 #"#d95f0e"
BORDER_COLOR= "black"; BORDER_WIDTH= 1.2

# --- SIZE CONTROLS ---
SIMPLIFY_M = 1        # <-- simplify geometry (meters). Try 50–150 for ~10MB EPS.
RASTERIZE  = "roads"   # choices: None, "roads", "all"  (rasterizes heavy layers in EPS)
PNG_DPI    = 300

# Optional extra compression of paths in Matplotlib (helps a bit)
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

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roads  = read_lines_project(ROADS_PATH)
    rails  = read_lines_project(RAILS_PATH)
    conns  = read_lines_project(CONNS_PATH)
    border = read_lines_project(BORDER_PATH) if BORDER_PATH.exists() else None

    # Geometry simplification to keep EPS small
    roads = simplify_gdf(roads, SIMPLIFY_M)
    rails = simplify_gdf(rails, SIMPLIFY_M * 0.8)    # a touch less, keep rail crisper
    conns = simplify_gdf(conns, SIMPLIFY_M * 0.6)    # keep connections sharpest

    # Figure
    fig, ax = plt.subplots(figsize=(8.0, 9.6))  # width x (width*1.2)
    ax.set_axis_off()
    # ax.set_title("Germany: Roads, Rail & Direct Connections", pad=10)

    # Draw order: roads → rail → connections → border
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
        border.plot(ax=ax, color=BORDER_COLOR, linewidth=BORDER_WIDTH, zorder=5)
    # Legend (upper-left)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ROADS_COLOR, lw=RAIL_WIDTH*4, label="Road"),
        Line2D([0], [0], color=RAIL_COLOR,  lw=RAIL_WIDTH*4,  label="Rail"),
        Line2D([0], [0], color=CONN_COLOR,  lw=CONN_WIDTH*4,  label="Connection"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_EPS, format="eps", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved EPS: {OUT_EPS}")

if __name__ == "__main__":
    main()
