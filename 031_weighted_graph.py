import json
from pathlib import Path
from statistics import mean
import csv
import math
import geopandas as gpd
from shapely.geometry import LineString
import datetime
from collections import defaultdict

from functions_weighted_graph import *

##############################
# inputs
##############################
UPDATE = 0
BASE_DIR = Path(__file__).resolve().parent

ROAD_JSON = BASE_DIR / 'input_files' / "germany_all_road_edges.json"
RAIL_JSON = BASE_DIR / 'input_files' / "germany_rail_edges.json"
POIS_GPKG = BASE_DIR / 'input_files' / "final_pois.gpkg"

Q_TEUS = 80.0  # number of TEUs

OUTPUT_DIR = BASE_DIR / 'graphs'
OUTPUT_FILENAME = f"graph_q_{int(Q_TEUS)}_oneway.json"
OUTPUT_EDGES_JSON = OUTPUT_DIR / OUTPUT_FILENAME
INPUT_EDGES_JSON = BASE_DIR / 'graphs' / 'graph_q_80.json'

EXPORT_CSV = True
OUTPUT_EDGES_CSV = OUTPUT_EDGES_JSON.with_suffix(".csv")

##############################
# parameters (only informative,
# in case of value change, change in functions_weighted_graph.py)
##############################
KMH_MOTORWAY = 80   # 76 km/h
KMH_SECONDARY = 60   # 57 km/h
KMH_RAIL = 55.0

G_PER_TKM_ROAD = 129.5
G_PER_TKM_RAIL = 24.0
TONS_PER_TEU = 11.5
G_PER_HOUR_TRANSFER = 0

SECONDS_PER_TEU_CONNECTION = 100.0
MINUTES_FIXED_CONNECTION = 111.0

# Costs €/km·TEU
ROAD_COST_PER_KM_PER_TEU = (1.99 + 0.30)/2
RAIL_COST_PER_KM_PER_TEU = 0.05
CONNECTION_COST_PER_TEU = 50.0


##############################
# Main
##############################
if __name__ == "__main__":
    if UPDATE == 0:
        print('Building new graph...')
        road_edges = load_edges(ROAD_JSON)
        rail_edges = load_edges(RAIL_JSON)
        road_graph = build_graph_with_mode(road_edges, mode="road")
        rail_graph = build_graph_with_mode(rail_edges, mode="rail")
        graph = merge_graphs(road_graph, rail_graph)
        added, skipped, pairs = add_poi_connections(
            graph,
            road_graph,
            rail_graph,
            pois_gpkg=POIS_GPKG,
            q_teus=Q_TEUS
        )
        print("Connections added:", added, "skipped:", skipped)

    else:
        print('Loading existing graph and refreshing connection lengths...')
        with open(INPUT_EDGES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = defaultdict(list)
        for e in data:
            u = (e["u"]["mode"], float(e["u"]["lon"]), float(e["u"]["lat"]))
            v = (e["v"]["mode"], float(e["v"]["lon"]), float(e["v"]["lat"]))
            w = float(e["length_m"]); m = e["mode"]
            if m == "connection":
                w = 0
            graph[u].append((v, w, m))
            graph[v].append((u, w, m))

    # Unique edges and weight calculation
    edges = []
    seen = set()
    for u, nbrs in graph.items():
        for v, length_m, mode in nbrs:
            key = frozenset((u, v))
            if key in seen:
                continue
            seen.add(key)
            road_mode = ""
            e_out = {
                "u": {"mode": u[0], "lon": u[1], "lat": u[2]},
                "v": {"mode": v[0], "lon": v[1], "lat": v[2]},
                "length_m": float(length_m),
                "mode": mode
            }
            e_out["time_h"]       = compute_time_hours(length_m, mode, Q_TEUS, road_mode)
            e_out["emissions_g"]  = compute_emissions_grams(length_m, mode, Q_TEUS)
            e_out["cost_eur"]     = compute_cost_eur(length_m, mode, Q_TEUS)
            edges.append(e_out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_edges_json(OUTPUT_EDGES_JSON, edges)
    if EXPORT_CSV:
        save_edges_csv(OUTPUT_EDGES_CSV, edges)
    print(f"Graph saved to {OUTPUT_EDGES_JSON} with {len(edges)} edges.")
    summarize_edges(edges)
