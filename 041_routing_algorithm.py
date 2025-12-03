import json
import math
import heapq
import time
from pathlib import Path
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from matplotlib.lines import Line2D
import datetime
import pandas as pd

from functions_routing_algorithm import *

##############################
# inputs (Modifyable parameters)
##############################
BASE_DIR = Path(__file__).resolve().parent

# 1) Number of TEUs to transport in the route (Q_TEUS)
Q_TEU = 20

GRAPH_DIR = BASE_DIR / "graphs" / f'graph_q_{Q_TEU}_oneway.json'

# 2) Start/goal points

locations_dic = {
    'Berlin': (13.315287220634708, 52.584036356012966),
    'Munich': (11.573904620929909, 48.13446772388894),
    'Frankfurt am Main': (8.656278, 50.102292),
    'Stuttgart': (9.185551, 48.787472),
    'Hamburg': (10.000624, 53.447532),
    'Freiburg im Breisgau': (7.837422, 47.993896),
    'Birkenfeld':(7.175697, 49.645312),
    'Salzwedel': (11.171322, 52.839897),
    'Wolfsburg' : (10.778848, 52.416499)
}# (lon, lat)

# CHANGE HERE
start_location = 'Hamburg'
goal_location   = 'Stuttgart'

START_POINT = locations_dic[start_location]
GOAL_POINT  = locations_dic[goal_location]

# 3) Criteria used by Dijkstra: 'length' | 'cost' | 'time' | 'emissions'
WEIGHT_ATTR = 'length'

##############################
# Main
##############################
if __name__ == "__main__":
    # 1) Load graph
    print('Section 1:' , datetime.datetime.now(datetime.timezone.utc))
    graph = load_graph_from_edge_list(GRAPH_DIR)

    # 2) Route report
    print('Route details:')
    print('Origin:', start_location)
    print('Destination:', goal_location)
    print('Optimization criteria:', WEIGHT_ATTR)
    print('TEUs:', Q_TEU)

    # 3) Equivalent nodes for start and goal points
    all_nodes = set(graph.keys())
    for u, nbrs in graph.items():
        for e in nbrs:
            all_nodes.add(e["v"])

    start_node, _ = find_nearest_node(all_nodes, START_POINT)
    goal_node,  _ = find_nearest_node(all_nodes, GOAL_POINT)

    # 4) Dijkstra's algorithm execution
    t0 = time.perf_counter()
    print('Start Dijkstra with weight =', WEIGHT_ATTR)
    path, total_weight = dijkstra(graph, start_node, goal_node, WEIGHT_ATTR)
    t1 = time.perf_counter()

    # 5) Final path summary
    path_length_m = sum_path_metric(graph, path, criteria="length")

    units = {'length':'m','time':'h','emissions':'g','cost':'€'}[WEIGHT_ATTR]
    print(f"Optimized total {WEIGHT_ATTR}: {total_weight:.6f} ({units})")
    print(f"True path length (m): {path_length_m:.2f}")
    print(f"Number of path points: {len(path)}")
    print(f"Dijkstra runtime: {t1 - t0:.3f} s\n")

    segments = summarize_path_by_mode(graph, path, criteria=WEIGHT_ATTR)
    print_mode_summary(segments)

    # 6) Path plotting and saving
    print('Section 5:', datetime.datetime.now(datetime.timezone.utc))

    # Save path into .geojson file
    save_path_geojson(path, path_length_m, BASE_DIR / 'solutions' / f"o_{start_location}_d_{goal_location}_q_{Q_TEU}_a_{WEIGHT_ATTR}.geojson")

    # Save information to excel
    write_run_to_excel(
        excel_path=BASE_DIR / "solutions" / "results.xlsx",
        graph=graph,
        path=path,
        start_location=start_location,
        goal_location=goal_location,
        q_teu=Q_TEU,
        weight_attr=WEIGHT_ATTR,
        dijkstra_runtime_s=(t1 - t0)
    )
    print(f"Saved: "+ str(BASE_DIR / "solutions" / "results.xlsx"))

    # Plot and save solution
    out_png = plot_route_with_border_final(
        graph=graph,
        path=path,
        start_point=START_POINT,
        goal_point=GOAL_POINT,
        start_name=start_location,
        goal_name=goal_location,
        weight_attr=WEIGHT_ATTR,
        q_teu=Q_TEU,
        base_dir=BASE_DIR,
    )
    print(f"Saved: {out_png}")