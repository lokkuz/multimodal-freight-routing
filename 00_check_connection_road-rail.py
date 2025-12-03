import geopandas as gpd
import numpy as np
from pathlib import Path

GPKG = Path("graphs/multimodal_network.gpkg")
edges = gpd.read_file(GPKG, layer="network_edges")
edges["mode"] = edges["mode"].str.lower()
edges["u"] = edges["u"].astype(int); edges["v"] = edges["v"].astype(int)

roads = edges[edges["mode"]=="road"]
rails = edges[edges["mode"]=="rail"]
conns = edges[edges["mode"]=="connection"]

road_nodes = set(np.r_[roads["u"].to_numpy(), roads["v"].to_numpy()])
rail_nodes = set(np.r_[rails["u"].to_numpy(), rails["v"].to_numpy()])

def classify_end(u, v):
    u_is_road = u in road_nodes
    u_is_rail = u in rail_nodes
    v_is_road = v in road_nodes
    v_is_rail = v in rail_nodes
    return u_is_road, u_is_rail, v_is_road, v_is_rail

ok = 0; swapped = 0; bad = 0
for _, r in conns.iterrows():
    u, v = int(r["u"]), int(r["v"])
    uR,uT,vR,vT = classify_end(u,v)  # R=road, T=rail
    # want exactly one rail end and exactly one road end
    if (uR or vR) and (uT or vT) and not (uT and vT) and not (uR and vR):
        ok += 1
        if uT and vR:
            swapped += 1  # rail on u, road on v (rare)
    else:
        bad += 1

print(f"Connections OK (one road end & one rail end): {ok} / {len(conns)}")
print(f"  …of which rail-on-u (swapped orientation): {swapped}")
print(f"Connections that violate the rule: {bad}")
