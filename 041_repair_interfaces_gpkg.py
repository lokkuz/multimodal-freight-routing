#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair a multimodal network GPKG so every connection edge is truly connected:
- Each connection endpoint is forced to share the EXACT node id with a rail node (one end)
  and a road node (the other end).
- If a suitable node is not nearby, the script splits the closest rail/road edge to create one.

Inputs (EPSG:25832 expected):
  graphs/multimodal_network.gpkg
    - layer 'network_edges' with columns: u (int), v (int), mode in {'road','rail','connection'}, geometry (LineString)
    - layer 'network_nodes' with columns: node_id (int), geometry (Point)

Outputs:
  graphs/multimodal_network_fixed.gpkg   (edges + nodes repaired)
"""

from pathlib import Path
import sys
import warnings
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import split as shp_split

# ----------- Config -----------
BASE = Path(".").resolve()
IN_GPKG  = BASE / "graphs" / "multimodal_network.gpkg"
OUT_GPKG = BASE / "graphs" / "multimodal_network_fixed.gpkg"
EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"

NET_EPSG = 25832   # meters

# Node matching & splitting tolerances (meters)
NODE_MATCH_TOL = 1.5   # snap a connection endpoint to an existing node if within this
EDGE_SPLIT_TOL = 15.0  # if no node close enough, split the closest edge if within this

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None


def load_net():
    if not IN_GPKG.exists():
        print(f"ERROR: not found: {IN_GPKG}", file=sys.stderr); sys.exit(1)
    edges = gpd.read_file(IN_GPKG, layer=EDGES_LAYER)
    nodes = gpd.read_file(IN_GPKG, layer=NODES_LAYER)
    if edges.crs is None: edges = edges.set_crs(NET_EPSG, allow_override=True)
    if nodes.crs is None: nodes = nodes.set_crs(NET_EPSG, allow_override=True)
    if edges.crs.to_epsg() != NET_EPSG: edges = edges.to_crs(NET_EPSG)
    if nodes.crs.to_epsg() != NET_EPSG: nodes = nodes.to_crs(NET_EPSG)
    edges["u"] = edges["u"].astype(int); edges["v"] = edges["v"].astype(int)
    edges["mode"] = edges["mode"].astype(str).str.lower()
    return edges, nodes


def kdtree_from_nodes(nodes_subset):
    ids = nodes_subset["node_id"].astype(int).to_numpy()
    arr = np.vstack([nodes_subset.geometry.x.to_numpy(), nodes_subset.geometry.y.to_numpy()]).T
    if KDTree is not None and len(ids) > 0:
        return KDTree(arr), ids, arr
    return None, ids, arr


def nearest_node_id(tree, ids, arr, xy):
    q = np.array(xy, float)
    if tree is not None and len(ids) > 0:
        dist, idx = tree.query(q)
        return int(ids[int(idx)]), float(dist)
    if len(ids) == 0:
        return None, float("inf")
    d2 = np.sum((arr - q) ** 2, axis=1)
    i = int(np.argmin(d2))
    return int(ids[i]), float(np.sqrt(d2[i]))


def build_spatial_index(gdf):
    _ = gdf.sindex
    return gdf


def nearest_edge_point(pt: Point, lines_gdf: gpd.GeoDataFrame, tol: float):
    """
    Return (row_index, projected_point, distance) to the closest line within tol (bbox prefilter).
    """
    bbox = pt.buffer(tol).bounds
    cand = list(lines_gdf.sindex.intersection(bbox))
    best = (None, None, float("inf"))
    for i in cand:
        geom = lines_gdf.geometry.iloc[i]
        proj = geom.project(pt)
        p2 = geom.interpolate(proj)
        d = p2.distance(pt)
        if d < best[2]:
            best = (i, p2, d)
    if best[0] is None or best[2] > tol:
        return None, None, None
    return best


def split_one_edge(edges_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, row_idx: int, point_on_edge: Point):
    """
    Split the edge at row_idx at 'point_on_edge'.
    - Removes the original edge row
    - Appends two new edges (u->new, new->v)
    - Adds a new node at the split point
    Returns updated (edges_gdf, nodes_gdf, new_node_id)
    """
    row = edges_gdf.iloc[row_idx]
    geom: LineString = row.geometry
    if geom is None or geom.is_empty:
        return edges_gdf, nodes_gdf, None

    # Perform split
    parts = shp_split(geom, MultiPoint([point_on_edge]))
    parts = [p for p in parts.geoms if isinstance(p, LineString) and p.length > 0]
    if len(parts) != 2:
        # Degenerate split; bail out
        return edges_gdf, nodes_gdf, None

    # Create a new node id at the split point
    new_id = int(nodes_gdf["node_id"].max() + 1 if len(nodes_gdf) else 0)
    nodes_gdf = gpd.GeoDataFrame(pd.concat([
        nodes_gdf,
        gpd.GeoDataFrame({"node_id":[new_id], "geometry":[Point(point_on_edge.x, point_on_edge.y)]},
                         geometry="geometry", crs=nodes_gdf.crs)
    ], ignore_index=True), geometry="geometry", crs=nodes_gdf.crs)

    # Original endpoints (assumed consistent with u/v)
    u0 = int(row["u"]); v0 = int(row["v"])
    # Build two new edges: (u0 -> new_id) and (new_id -> v0)
    # Determine orientation by comparing part endpoints to original u/v coordinates
    ux, uy = list(geom.coords)[0]; vx, vy = list(geom.coords)[-1]
    u_pt = Point(ux, uy); v_pt = Point(vx, vy)

    def edge_from_part(part_geom, start_id, end_id):
        return {**row.drop(labels=["geometry"]).to_dict(), "u": int(start_id), "v": int(end_id), "geometry": part_geom}

    # Identify which part touches u0 and which touches v0
    pA, pB = parts
    if Point(list(pA.coords)[0]).distance(u_pt) + Point(list(pA.coords)[-1]).distance(Point(point_on_edge)) < \
       Point(list(pB.coords)[0]).distance(u_pt) + Point(list(pB.coords)[-1]).distance(Point(point_on_edge)):
        part_u = pA; part_v = pB
    else:
        part_u = pB; part_v = pA

    row_u = edge_from_part(part_u, u0, new_id)
    row_v = edge_from_part(part_v, new_id, v0)

    # Rebuild edges_gdf: drop old row, append two
    edges_gdf = edges_gdf.drop(edges_gdf.index[row_idx]).reset_index(drop=True)
    edges_gdf = gpd.GeoDataFrame(pd.concat([edges_gdf,
                                            gpd.GeoDataFrame([row_u, row_v], geometry="geometry", crs=edges_gdf.crs)],
                                           ignore_index=True),
                                 geometry="geometry", crs=edges_gdf.crs)
    return edges_gdf, nodes_gdf, new_id


# ---------- Main repair ----------
def main():
    import pandas as pd

    edges, nodes = load_net()

    # Split by mode
    roads = edges[edges["mode"]=="road"].reset_index(drop=True)
    rails = edges[edges["mode"]=="rail"].reset_index(drop=True)
    conns = edges[edges["mode"]=="connection"].reset_index(drop=True)

    # Spatial indices for edges
    roads = build_spatial_index(roads)
    rails = build_spatial_index(rails)

    # KD-trees for existing ROAD and RAIL nodes
    road_node_ids = np.unique(np.concatenate([roads["u"].to_numpy(), roads["v"].to_numpy()])) if len(roads) else np.array([], int)
    rail_node_ids = np.unique(np.concatenate([rails["u"].to_numpy(), rails["v"].to_numpy()])) if len(rails) else np.array([], int)

    road_nodes = nodes[nodes["node_id"].isin(road_node_ids)].copy()
    rail_nodes = nodes[nodes["node_id"].isin(rail_node_ids)].copy()

    tree_road, ids_road, arr_road = kdtree_from_nodes(road_nodes)
    tree_rail, ids_rail, arr_rail = kdtree_from_nodes(rail_nodes)

    if len(conns) == 0:
        print("No connection edges found; nothing to repair.")
        sys.exit(0)

    def end_pts(ls: LineString):
        c = list(ls.coords); return (c[0][0], c[0][1]), (c[-1][0], c[-1][1])

    # Helpers to get nearest node and to split if needed
    def attach_endpoint_to_layer(xy, target_layer: str):
        """
        Returns (node_id, snapped_xy, action) where action in {"matched_node","split_edge","failed"}.
        May update roads/rails/nodes globals when splitting.
        """
        nonlocal roads, rails, nodes, road_nodes, rail_nodes, tree_road, tree_rail, ids_road, ids_rail, arr_road, arr_rail

        if target_layer == "road":
            nid, dist = nearest_node_id(tree_road, ids_road, arr_road, xy)
            if (nid is not None) and (dist <= NODE_MATCH_TOL):
                pt = road_nodes.loc[road_nodes["node_id"]==nid, "geometry"].values[0]
                return int(nid), (float(pt.x), float(pt.y)), "matched_node"

            # else try split nearest road edge
            idx, proj_pt, d = nearest_edge_point(Point(xy[0], xy[1]), roads, EDGE_SPLIT_TOL)
            if idx is None:
                return None, xy, "failed"
            roads, nodes, new_id = split_one_edge(roads, nodes, idx, proj_pt)
            # refresh road_nodes cache & KDTree
            road_node_ids = np.unique(np.concatenate([roads["u"].to_numpy(), roads["v"].to_numpy()])) if len(roads) else np.array([], int)
            road_nodes = nodes[nodes["node_id"].isin(road_node_ids)].copy()
            tree_road, ids_road, arr_road = kdtree_from_nodes(road_nodes)
            pt = nodes.loc[nodes["node_id"]==new_id, "geometry"].values[0]
            return int(new_id), (float(pt.x), float(pt.y)), "split_edge"

        else:  # target_layer == "rail"
            nid, dist = nearest_node_id(tree_rail, ids_rail, arr_rail, xy)
            if (nid is not None) and (dist <= NODE_MATCH_TOL):
                pt = rail_nodes.loc[rail_nodes["node_id"]==nid, "geometry"].values[0]
                return int(nid), (float(pt.x), float(pt.y)), "matched_node"

            idx, proj_pt, d = nearest_edge_point(Point(xy[0], xy[1]), rails, EDGE_SPLIT_TOL)
            if idx is None:
                return None, xy, "failed"
            rails, nodes, new_id = split_one_edge(rails, nodes, idx, proj_pt)
            rail_node_ids = np.unique(np.concatenate([rails["u"].to_numpy(), rails["v"].to_numpy()])) if len(rails) else np.array([], int)
            rail_nodes = nodes[nodes["node_id"].isin(rail_node_ids)].copy()
            tree_rail, ids_rail, arr_rail = kdtree_from_nodes(rail_nodes)
            pt = nodes.loc[nodes["node_id"]==new_id, "geometry"].values[0]
            return int(new_id), (float(pt.x), float(pt.y)), "split_edge"

    # Decide per-connection which end should be ROAD vs RAIL (minimize sum distances)
    def best_assignment(a_xy, b_xy):
        # distances to nearest road/rail nodes
        a_rnid, a_rd = nearest_node_id(tree_road, ids_road, arr_road, a_xy) if len(ids_road) else (None, float("inf"))
        a_tnid, a_td = nearest_node_id(tree_rail, ids_rail, arr_rail, a_xy) if len(ids_rail) else (None, float("inf"))
        b_rnid, b_rd = nearest_node_id(tree_road, ids_road, arr_road, b_xy) if len(ids_road) else (None, float("inf"))
        b_tnid, b_td = nearest_node_id(tree_rail, ids_rail, arr_rail, b_xy) if len(ids_rail) else (None, float("inf"))

        # candidates: (A->road,B->rail) vs (A->rail,B->road)
        cand1 = (a_rd + b_td, ("road","rail"))
        cand2 = (a_td + b_rd, ("rail","road"))
        return (cand1 if cand1[0] <= cand2[0] else cand2)[1]

    # Repair loop
    import pandas as pd
    fixed_rows = []
    splits_done = 0
    matched_nodes = 0
    failed_ends = 0
    fully_fixed_connections = 0

    for _, r in conns.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            fixed_rows.append(r); continue
        (ax, ay), (bx, by) = end_pts(geom)

        # Decide which end should attach to road vs rail
        A_to, B_to = best_assignment((ax, ay), (bx, by))

        # Attach A
        A_id, A_xy, A_action = attach_endpoint_to_layer((ax, ay), A_to)
        if A_action == "failed":
            # Try the other type as fallback
            A_to_alt = "rail" if A_to == "road" else "road"
            A_id, A_xy, A_action = attach_endpoint_to_layer((ax, ay), A_to_alt)

        # Attach B
        B_id, B_xy, B_action = attach_endpoint_to_layer((bx, by), B_to)
        if B_action == "failed":
            B_to_alt = "rail" if B_to == "road" else "road"
            B_id, B_xy, B_action = attach_endpoint_to_layer((bx, by), B_to_alt)

        matched_nodes += int(A_action == "matched_node") + int(B_action == "matched_node")
        splits_done  += int(A_action == "split_edge")   + int(B_action == "split_edge")
        failed_ends  += int(A_action == "failed")       + int(B_action == "failed")

        # Rebuild connection row (update u/v and geometry endpoints)
        coords = list(geom.coords)
        if A_id is not None:
            coords[0] = (A_xy[0], A_xy[1])
        if B_id is not None:
            coords[-1] = (B_xy[0], B_xy[1])

        new_geom = LineString(coords)
        r2 = r.copy()
        if A_id is not None: r2["u"] = int(A_id)
        if B_id is not None: r2["v"] = int(B_id)
        r2.geometry = new_geom
        fixed_rows.append(r2)

        if (A_id is not None) and (B_id is not None):
            fully_fixed_connections += 1

    # Reassemble repaired edges
    edges_fixed = pd.concat([
        roads,
        rails,
        gpd.GeoDataFrame(fixed_rows, geometry="geometry", crs=edges.crs)
    ], ignore_index=True)

    # Recompute some diagnostics
    acc = edges_fixed[edges_fixed["mode"].isin(["road","connection"])]
    rail = edges_fixed[edges_fixed["mode"]=="rail"]
    acc_nodes = set(np.concatenate([acc["u"].to_numpy(), acc["v"].to_numpy()])) if len(acc) else set()
    rail_nodes = set(np.concatenate([rail["u"].to_numpy(), rail["v"].to_numpy()])) if len(rail) else set()
    iface = acc_nodes & rail_nodes

    # Write out
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    edges_fixed.to_file(OUT_GPKG, layer=EDGES_LAYER, driver="GPKG")
    nodes.to_file(OUT_GPKG, layer=NODES_LAYER, driver="GPKG")

    print(f"✅ Wrote repaired network: {OUT_GPKG}")
    print(f"   Connections processed: {len(conns)} | fully fixed (both ends linked): {fully_fixed_connections}")
    print(f"   Node matches reused: {matched_nodes} | Edges split to create nodes: {splits_done} | Endpoints still failing: {failed_ends}")
    print(f"   Interface nodes after repair: {len(iface)} (was {len(acc_nodes & rail_nodes)})")
    print("   Hint: rerun your audit and routing on the *_fixed.gpkg file.")

if __name__ == "__main__":
    # lazy import to keep top tidy
    import pandas as pd
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)
