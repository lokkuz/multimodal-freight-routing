#!/usr/bin/env python3
"""
Create the multimodal graph (road + rail + connections) and export it.
No CLI, no switches — just run this file in PyCharm.

Inputs are expected under ./input_files/
Outputs are written to ./graphs/

Requires: functions_weighted_graph.py in the same folder.
"""

from pathlib import Path
from collections import defaultdict
import json
import sys

# ---- user knobs (edit if you like) -----------------------------------------
Q_TEUS = 20.0                     # shipment size in TEUs
EXPORT_CSV = True                 # also write a CSV next to the JSON

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_files"
GRAPHS_DIR = BASE_DIR / "graphs"

ROAD_JSON = INPUT_DIR / "germany_all_road_edges.json"
RAIL_JSON = INPUT_DIR / "germany_rail_edges.json"
POIS_GPKG = INPUT_DIR / "final_pois.gpkg"

OUTPUT_EDGES_JSON = GRAPHS_DIR / f"graph_q_{int(Q_TEUS)}_oneway.json"
# ----------------------------------------------------------------------------

try:
    from functions_weighted_graph import (
        load_edges,
        build_graph_with_mode,
        merge_graphs,
        add_poi_connections,
        compute_time_hours,
        compute_emissions_grams,
        compute_cost_eur,
        save_edges_json,
        save_edges_csv,
        summarize_edges,
    )
except ImportError as e:
    print("[error] Couldn't import functions_weighted_graph.py")
    print("        Make sure this file sits next to create_graph_pycharm.py")
    raise


def build_graph_from_inputs(road_json: Path, rail_json: Path, pois_gpkg: Path, q_teus: float):
    """Create neighbor map from road/rail edges and add transfer connections via POIs."""
    road_edges = load_edges(road_json)
    rail_edges = load_edges(rail_json)

    road_graph = build_graph_with_mode(road_edges, mode="road")
    rail_graph = build_graph_with_mode(rail_edges, mode="rail")
    graph = merge_graphs(road_graph, rail_graph)

    added, skipped, _ = add_poi_connections(
        graph,
        road_graph,
        rail_graph,
        pois_gpkg=pois_gpkg,
        q_teus=q_teus,
    )
    print(f"[create] connection links → added={added}, skipped={skipped}")
    return graph


def graph_to_weighted_edges(graph, q_teus: float):
    """Turn neighbor map into unique (undirected) weighted edges with metrics."""
    edges = []
    seen = set()
    for u, nbrs in graph.items():
        for v, length_m, mode in nbrs:
            key = frozenset((u, v))
            if key in seen:
                continue
            seen.add(key)
            road_mode = ""  # pass subtype if your model distinguishes motorway/secondary
            e_out = {
                "u": {"mode": u[0], "lon": u[1], "lat": u[2]},
                "v": {"mode": v[0], "lon": v[1], "lat": v[2]},
                "length_m": float(length_m),
                "mode": mode,
            }
            e_out["time_h"]      = compute_time_hours(length_m, mode, q_teus, road_mode)
            e_out["emissions_g"] = compute_emissions_grams(length_m, mode, q_teus)
            e_out["cost_eur"]    = compute_cost_eur(length_m, mode, q_teus)
            edges.append(e_out)
    return edges


if __name__ == "__main__":
    # Basic input checks
    missing = [p for p in [ROAD_JSON, RAIL_JSON, POIS_GPKG] if not p.exists()]
    if missing:
        print("[create] Missing required input files:")
        for p in missing:
            print("  -", p)
        sys.exit(2)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    print("[create] Building graph from inputs…")
    graph = build_graph_from_inputs(ROAD_JSON, RAIL_JSON, POIS_GPKG, Q_TEUS)

    print("[create] Computing weights…")
    edges = graph_to_weighted_edges(graph, Q_TEUS)

    print(f"[create] Saving → {OUTPUT_EDGES_JSON}")
    save_edges_json(OUTPUT_EDGES_JSON, edges)
    if EXPORT_CSV:
        save_edges_csv(OUTPUT_EDGES_JSON.with_suffix(".csv"), edges)

    print(f"[create] Done. {len(edges)} edges written.")
    summarize_edges(edges)
