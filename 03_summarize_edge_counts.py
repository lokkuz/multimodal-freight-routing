#!/usr/bin/env python3
# pip install geopandas

from pathlib import Path
import geopandas as gpd

# --- Paths (edit if your layout differs)
BASE_DIR = Path(__file__).resolve().parent
ROADS = BASE_DIR / "input_files" / "derived" / "germany_all_road_edges.geojson"
RAILS = BASE_DIR / "input_files" / "derived" / "germany_rail_edges.geojson"
CONNS = BASE_DIR / "graphs" / "road_rail_connection_edges.geojson"

def count_edges(path: Path) -> int:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return 0
    try:
        gdf = gpd.read_file(path)
        return len(gdf)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return 0

def main():
    n_roads = count_edges(ROADS)
    n_rails = count_edges(RAILS)
    n_conns = count_edges(CONNS)
    total = n_roads + n_rails + n_conns

    print(f"Road edges:       {n_roads:,}")
    print(f"Rail edges:       {n_rails:,}")
    print(f"Connection edges: {n_conns:,}")
    print("-" * 32)
    print(f"Total edges:      {total:,}")

if __name__ == "__main__":
    main()
