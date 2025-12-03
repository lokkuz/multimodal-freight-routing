#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_fix_implicit_interfaces.py (defaults to your weighted graph)

Split nodes that are incident to BOTH ROAD and RAIL so free mode switches
can't happen. Safe for weighted networks (keeps all weight columns).

Defaults:
  --in  ./graphs/multimodal_network_with_weights.gpkg
  --out ./graphs/multimodal_network_with_weights_sanitized.gpkg
No connection edges are added by default; nothing is dropped by default.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"
EPSG_METRIC = 25832

BASE = Path(".").resolve()
DEF_IN  = BASE / "graphs" / "multimodal_network_with_weights.gpkg"
DEF_OUT = BASE / "graphs" / "multimodal_network_with_weights_sanitized.gpkg"

# ---------- IO ----------
def load_layers(gpkg: Path):
    if not gpkg.exists():
        print(f"ERROR: Input not found: {gpkg}", file=sys.stderr); sys.exit(1)
    edges = gpd.read_file(gpkg, layer=EDGES_LAYER)
    nodes = gpd.read_file(gpkg, layer=NODES_LAYER)

    # Ensure CRS
    if edges.crs is None: edges = edges.set_crs(epsg=EPSG_METRIC, allow_override=True)
    if nodes.crs is None: nodes = nodes.set_crs(epsg=EPSG_METRIC, allow_override=True)
    if edges.crs.to_epsg() != EPSG_METRIC: edges = edges.to_crs(EPSG_METRIC)
    if nodes.crs.to_epsg() != EPSG_METRIC: nodes = nodes.to_crs(EPSG_METRIC)

    # Normalize essentials
    for col in ("u","v"):
        if col not in edges.columns:
            raise RuntimeError(f"Edges layer missing required column '{col}'.")
        edges[col] = edges[col].astype(int)
    if "mode" not in edges.columns:
        raise RuntimeError("Edges layer missing required column 'mode'.")
    edges["mode"] = edges["mode"].astype(str).str.lower()

    if "node_id" not in nodes.columns:
        raise RuntimeError("Nodes layer missing required column 'node_id'.")
    nodes["node_id"] = nodes["node_id"].astype(int)
    return edges, nodes

def recompute_lengths_in_place(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edges["length_m"] = edges.geometry.length.astype(float)
    return edges

# ---------- Detect mixed (road+rail) nodes ----------
def find_implicit_interfaces(edges: gpd.GeoDataFrame) -> np.ndarray:
    acc  = edges[edges["mode"].isin(["road"])]
    rail = edges[edges["mode"]=="rail"]
    acc_nodes  = np.unique(np.r_[acc["u"].to_numpy(int),  acc["v"].to_numpy(int)])  if len(acc)  else np.array([], int)
    rail_nodes = np.unique(np.r_[rail["u"].to_numpy(int), rail["v"].to_numpy(int)]) if len(rail) else np.array([], int)
    return np.intersect1d(acc_nodes, rail_nodes, assume_unique=False)

# ---------- Split & (optionally) add/drop connections ----------
def split_mixed_nodes(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    add_connections: bool = False,
    drop_existing_conns: bool = False,
    conn_time_min: float = 0.0,
    conn_cost_eur: float = 0.0,
    conn_co2e_kg: float = 0.0
):
    mixed = find_implicit_interfaces(edges)
    print(f"Mixed road+rail nodes (before): {len(mixed)}")
    if len(mixed) == 0:
        return edges, nodes, 0

    max_id = int(nodes["node_id"].max()) if len(nodes) else 0
    rail_new_ids = {}
    new_nodes_rows = []

    for n in mixed:
        max_id += 1
        rail_new_ids[n] = max_id
        pt = nodes.loc[nodes["node_id"]==n, "geometry"].iloc[0]
        new_nodes_rows.append({"node_id": max_id, "geometry": pt})

    if new_nodes_rows:
        nodes = gpd.GeoDataFrame(
            pd.concat([nodes, gpd.GeoDataFrame(new_nodes_rows, geometry="geometry", crs=nodes.crs)], ignore_index=True),
            geometry="geometry", crs=nodes.crs
        )

    # Rewire RAIL edges touching the mixed nodes to the new rail-only node
    m_rail = edges["mode"]=="rail"
    m_u = edges["u"].isin(mixed) & m_rail
    if m_u.any():
        edges.loc[m_u, "u"] = edges.loc[m_u, "u"].map(lambda x: rail_new_ids.get(int(x), int(x)))
    m_v = edges["v"].isin(mixed) & m_rail
    if m_v.any():
        edges.loc[m_v, "v"] = edges.loc[m_v, "v"].map(lambda x: rail_new_ids.get(int(x), int(x)))

    # Optional: add explicit connection edges (OFF by default)
    if add_connections:
        edge_cols = list(edges.columns)
        has_time  = "time_min" in edge_cols
        has_cost  = "cost_eur" in edge_cols
        has_emis  = "co2e_kg" in edge_cols
        has_wtime = "weight_time" in edge_cols
        has_wcost = "weight_cost" in edge_cols
        has_wemis = "weight_emission" in edge_cols

        node_xy = {int(r.node_id): (float(r.geometry.x), float(r.geometry.y)) for r in nodes.itertuples(index=False)}
        new_rows = []
        for old_id, new_id in rail_new_ids.items():
            xy = node_xy[old_id]
            geom = LineString([xy, xy])  # zero-length
            row = {c: None for c in edge_cols}
            row.update({"u": int(old_id), "v": int(new_id), "mode": "connection", "geometry": geom, "length_m": float(geom.length)})
            if has_time:  row["time_min"] = float(conn_time_min)
            if has_cost:  row["cost_eur"] = float(conn_cost_eur)
            if has_emis:  row["co2e_kg"]  = float(conn_co2e_kg)
            if has_wtime: row["weight_time"] = float(conn_time_min)
            if has_wcost: row["weight_cost"] = float(conn_cost_eur)
            if has_wemis: row["weight_emission"] = float(conn_co2e_kg)
            new_rows.append(row)
        if new_rows:
            edges = gpd.GeoDataFrame(
                pd.concat([edges, gpd.GeoDataFrame(new_rows, columns=edge_cols)], ignore_index=True),
                geometry="geometry", crs=edges.crs
            )

    # Optional: drop any existing connection edges touching either side
    if drop_existing_conns:
        iface_set = set(mixed) | set(rail_new_ids.values())
        before = len(edges)
        edges = edges[~((edges["mode"]=="connection") & (edges["u"].isin(iface_set) | edges["v"].isin(iface_set)))].copy()
        print(f"Dropped connection edges at split interfaces: {before - len(edges)}")

    edges = recompute_lengths_in_place(edges)

    remaining = find_implicit_interfaces(edges)
    print(f"Mixed road+rail nodes (after): {len(remaining)}")
    return edges, nodes, len(rail_new_ids)

# ---------- Write ----------
def write_layers(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, out_path: Path, overwrite: bool):
    if overwrite and out_path.exists(): out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_file(out_path, layer=EDGES_LAYER, driver="GPKG")
    nodes.to_file(out_path, layer=NODES_LAYER, driver="GPKG")
    print(f"✅ Wrote: {out_path}")
    print(f"   Edges: {len(edges):,}  Nodes: {len(nodes):,}")
    try:
        print(f"   Mode counts: {edges['mode'].value_counts().to_dict()}")
    except Exception:
        pass

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Split implicit road+rail nodes (weighted-safe). No connections added by default.")
    ap.add_argument("--in",  dest="in_gpkg",  type=str, default=str(DEF_IN),  help="Input GPKG (weighted ok)")
    ap.add_argument("--out", dest="out_gpkg", type=str, default=str(DEF_OUT), help="Output GPKG path")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")

    # Behavior toggles
    ap.add_argument("--drop-existing-conns", action="store_true",
                    help="Remove any existing 'connection' edges that touch split interfaces")
    ap.add_argument("--create-conns", action="store_true",
                    help="(Optional) Add explicit connection edges between split road/rail nodes")

    # If creating connection edges, set fixed weights (for weighted networks)
    ap.add_argument("--conn-fixed-time-min", type=float, default=0.0, help="Fixed minutes for created connections")
    ap.add_argument("--conn-fixed-cost-eur", type=float, default=0.0, help="Fixed € for created connections")
    ap.add_argument("--conn-fixed-emis-kg",  type=float, default=0.0, help="Fixed kg CO2e for created connections")

    args = ap.parse_args()

    edges, nodes = load_layers(Path(args.in_gpkg))
    edges2, nodes2, nfixed = split_mixed_nodes(
        edges, nodes,
        add_connections=args.create_conns,
        drop_existing_conns=args.drop_existing_conns,
        conn_time_min=float(args.conn_fixed_time_min),
        conn_cost_eur=float(args.conn_fixed_cost_eur),
        conn_co2e_kg=float(args.conn_fixed_emis_kg),
    )
    write_layers(edges2, nodes2, Path(args.out_gpkg), overwrite=args.overwrite)
    if nfixed:
        print(f"✔ Split {nfixed} implicit interface node(s) into road-only + rail-only.")
    else:
        print("No changes were necessary.")

if __name__ == "__main__":
    main()
