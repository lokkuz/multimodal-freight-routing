#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair connection edges so that each has exactly one road end and one rail end.
Also (optionally) enforce rail on v.

Input : graphs/multimodal_network.gpkg  (layers: network_edges, network_nodes)
Output: overwrites the same file by default (change OUT_GPKG to keep original)

Assumptions:
- CRS = EPSG:25832
- Edges layer columns: ['u','v','mode','geometry'] with mode in {'road','rail','connection'}
- Nodes layer columns: ['node_id','geometry']
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

IN_GPKG  = Path("graphs/multimodal_network.gpkg")
OUT_GPKG = IN_GPKG  # change to a new Path if you prefer not to overwrite
EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"
EPSG = 25832

# Tolerances (meters)
SNAP_TOL_FIX = 5.0   # max distance to accept snapping a bad endpoint to the other layer's nearest node
TIE_BREAK    = 0.50  # if rail/road distances are similar within this margin, prefer rail
ROUND = 6           # rounding for stable xy->id maps

# Enforce rail on v by flipping (if True)
FORCE_RAIL_ON_V = True

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

def kdtree_from_nodes(nodes_xy):
    arr = np.array(nodes_xy, float) if len(nodes_xy) else np.zeros((0,2))
    if KDTree is not None and len(arr):
        return KDTree(arr), arr
    return None, arr

def nearest_xy(tree, arr, qxy):
    q = np.array(qxy, float)
    if tree is not None and len(arr):
        dist, idx = tree.query(q)
        return float(dist), tuple(arr[int(idx)])
    if len(arr)==0:
        return float("inf"), None
    d2 = np.sum((arr - q)**2, axis=1)
    i = int(np.argmin(d2))
    return float(np.sqrt(d2[i])), tuple(arr[i])

def main():
    if not IN_GPKG.exists():
        print("ERROR: missing", IN_GPKG, file=sys.stderr); sys.exit(1)

    edges = gpd.read_file(IN_GPKG, layer=EDGES_LAYER)
    nodes = gpd.read_file(IN_GPKG, layer=NODES_LAYER)
    # normalize dtype/CRS
    if edges.crs is None: edges = edges.set_crs(EPSG, allow_override=True)
    if nodes.crs is None: nodes = nodes.set_crs(EPSG, allow_override=True)
    if edges.crs.to_epsg()!=EPSG: edges = edges.to_crs(EPSG)
    if nodes.crs.to_epsg()!=EPSG: nodes = nodes.to_crs(EPSG)
    edges["mode"] = edges["mode"].astype(str).str.lower()
    edges["u"] = edges["u"].astype(int); edges["v"] = edges["v"].astype(int)
    nodes["node_id"] = nodes["node_id"].astype(int)

    # Build node maps for existing rail/road nodes from edges (robust against orphan nodes)
    def node_maps_by_mode(mode_name):
        sub = edges[edges["mode"]==mode_name]
        ids = np.unique(np.r_[sub["u"].to_numpy(), sub["v"].to_numpy()]) if len(sub) else np.array([], int)
        sel = nodes[nodes["node_id"].isin(ids)].copy()
        id2pt = sel.set_index("node_id")["geometry"].to_dict()
        xy2id = {(round(float(pt.x),ROUND), round(float(pt.y),ROUND)): int(nid) for nid,pt in id2pt.items()}
        xy = [(float(pt.x), float(pt.y)) for pt in id2pt.values()]
        return id2pt, xy2id, xy

    road_id2pt, road_xy2id, road_xy = node_maps_by_mode("road")
    rail_id2pt, rail_xy2id, rail_xy = node_maps_by_mode("rail")

    road_tree, road_arr = kdtree_from_nodes(road_xy)
    rail_tree, rail_arr = kdtree_from_nodes(rail_xy)

    def is_road_node(nid): return nid in road_id2pt
    def is_rail_node(nid): return nid in rail_id2pt

    conns = edges[edges["mode"]=="connection"].copy()
    fixed_rows = []
    stats = {"ok":0, "fixed":0, "flipped":0, "could_not_fix":0, "made_road2rail":0, "made_rail2road":0}

    for _, r in conns.iterrows():
        ls: LineString = r.geometry
        if ls is None or ls.is_empty: 
            fixed_rows.append(r); continue
        coords = list(ls.coords)
        (x0,y0),(x1,y1) = coords[0], coords[-1]
        u = int(r["u"]); v = int(r["v"])
        u_isR, u_isT = is_road_node(u), is_rail_node(u)
        v_isR, v_isT = is_road_node(v), is_rail_node(v)

        def snap_to_other_layer(xy, want="rail"):
            if want=="rail":
                d, new_xy = nearest_xy(rail_tree, rail_arr, xy)
                if d <= SNAP_TOL_FIX and new_xy is not None:
                    nid = rail_xy2id.get((round(new_xy[0],ROUND), round(new_xy[1],ROUND)))
                    if nid is not None:
                        return int(nid), new_xy
            else:
                d, new_xy = nearest_xy(road_tree, road_arr, xy)
                if d <= SNAP_TOL_FIX and new_xy is not None:
                    nid = road_xy2id.get((round(new_xy[0],ROUND), round(new_xy[1],ROUND)))
                    if nid is not None:
                        return int(nid), new_xy
            return None, xy

        def classify_auto(xy0, xy1):
            dR0,_ = nearest_xy(rail_tree, rail_arr, xy0); dD0,_ = nearest_xy(road_tree, road_arr, xy0)
            dR1,_ = nearest_xy(rail_tree, rail_arr, xy1); dD1,_ = nearest_xy(road_tree, road_arr, xy1)
            s1 = dR0 + dD1  # u->rail, v->road
            s2 = dD0 + dR1  # u->road, v->rail
            return ("rail","road") if s1 <= s2 - TIE_BREAK else ("road","rail")

        # Case analysis
        if (u_isR or v_isR) and (u_isT or v_isT) and not (u_isR and v_isR) and not (u_isT and v_isT):
            # already one road & one rail
            stats["ok"] += 1
            # enforce rail on v if requested
            if FORCE_RAIL_ON_V and u_isT and not v_isT:
                # flip: swap u<->v and reverse geometry
                r["u"], r["v"] = v, u
                r.geometry = LineString(list(reversed(coords)))
                stats["flipped"] += 1
            fixed_rows.append(r)
            continue

        # Both road?
        if u_isR and v_isR:
            # push the far end to rail (prefer adjusting v)
            new_id, new_xy = snap_to_other_layer((x1,y1), want="rail")
            if new_id is None:
                # try u side
                new_id, new_xy = snap_to_other_layer((x0,y0), want="rail")
                if new_id is None:
                    stats["could_not_fix"] += 1
                    fixed_rows.append(r); continue
                # adjust u
                r["u"] = new_id; coords[0] = (float(new_xy[0]), float(new_xy[1]))
            else:
                # adjust v
                r["v"] = new_id; coords[-1] = (float(new_xy[0]), float(new_xy[1]))
            r.geometry = LineString(coords)
            stats["fixed"] += 1; stats["made_road2rail"] += 1

        # Both rail?
        elif u_isT and v_isT:
            # push the far end to road (prefer adjusting u)
            new_id, new_xy = snap_to_other_layer((x0,y0), want="road")
            if new_id is None:
                # try v side
                new_id, new_xy = snap_to_other_layer((x1,y1), want="road")
                if new_id is None:
                    stats["could_not_fix"] += 1
                    fixed_rows.append(r); continue
                # adjust v
                r["v"] = new_id; coords[-1] = (float(new_xy[0]), float(new_xy[1]))
            else:
                # adjust u
                r["u"] = new_id; coords[0] = (float(new_xy[0]), float(new_xy[1]))
            r.geometry = LineString(coords)
            stats["fixed"] += 1; stats["made_rail2road"] += 1

        else:
            # Neither endpoint recognized (should be rare) -> choose by min total distance
            prefer_u, prefer_v = classify_auto((x0,y0), (x1,y1))  # 'rail'/'road'
            if prefer_u == "rail":
                # make u rail, v road
                uid, uxy = snap_to_other_layer((x0,y0), want="rail")
                vid, vxy = snap_to_other_layer((x1,y1), want="road")
            else:
                uid, uxy = snap_to_other_layer((x0,y0), want="road")
                vid, vxy = snap_to_other_layer((x1,y1), want="rail")
            # accept if both found
            if uid is not None and vid is not None:
                r["u"] = int(uid); r["v"] = int(vid)
                coords[0] = (float(uxy[0]), float(uxy[1]))
                coords[-1] = (float(vxy[0]), float(vxy[1]))
                r.geometry = LineString(coords)
                stats["fixed"] += 1
            else:
                stats["could_not_fix"] += 1

        # enforce rail on v if requested
        u = int(r["u"]); v = int(r["v"])
        if FORCE_RAIL_ON_V and (u in rail_id2pt) and (v not in rail_id2pt):
            # flip to put rail on v
            r["u"], r["v"] = v, u
            r.geometry = LineString(list(reversed(coords)))
            stats["flipped"] += 1

        fixed_rows.append(r)

    # Merge repaired connections back
    not_conn = edges[edges["mode"]!="connection"].copy()
    edges_fixed = pd.concat([not_conn, gpd.GeoDataFrame(fixed_rows, geometry="geometry", crs=edges.crs)],
                            ignore_index=True)
    edges_fixed["u"] = edges_fixed["u"].astype(int)
    edges_fixed["v"] = edges_fixed["v"].astype(int)

    # Write out
    if OUT_GPKG.exists(): OUT_GPKG.unlink()
    edges_fixed.to_file(OUT_GPKG, layer=EDGES_LAYER, driver="GPKG")
    nodes.to_file(OUT_GPKG, layer=NODES_LAYER, driver="GPKG")

    # Sanity report
    edges = edges_fixed
    roads = edges[edges["mode"]=="road"]
    rails = edges[edges["mode"]=="rail"]
    conns = edges[edges["mode"]=="connection"]
    road_nodes = set(np.r_[roads["u"], roads["v"]]); rail_nodes = set(np.r_[rails["u"], rails["v"]])

    ok = 0; swapped = 0; bad = 0
    for _, r in conns.iterrows():
        u, v = int(r["u"]), int(r["v"])
        uR, uT = u in road_nodes, u in rail_nodes
        vR, vT = v in road_nodes, v in rail_nodes
        if (uR or vR) and (uT or vT) and not (uR and vR) and not (uT and vT):
            ok += 1
            if uT and not vT: swapped += 1
        else:
            bad += 1

    print("✅ Wrote:", OUT_GPKG)
    print(f"Connections OK (one road end & one rail end): {ok} / {len(conns)}")
    print(f"  …rail-on-u (orientation swapped): {swapped}")
    print(f"Connections still violating: {bad}")
    print(f"Repairs: fixed={stats['fixed']}, could_not_fix={stats['could_not_fix']}, flips={stats['flipped']},"
          f" made road→rail={stats['made_road2rail']}, made rail→road={stats['made_rail2road']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr); sys.exit(2)
