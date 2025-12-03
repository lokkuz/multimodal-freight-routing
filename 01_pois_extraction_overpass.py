import time
import requests
from requests.exceptions import RequestException
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

# -------------------------------
# Overpass helpers (robust retries)
# -------------------------------
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

HEADERS = {
    "User-Agent": "final_multimodal_routing/1.0 (+mailto:you@example.com)"
}

def query_overpass(query: str, timeout=180, max_retries=4, backoff_base=1.6):
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    endpoint,
                    data={"data": query},
                    headers=HEADERS,
                    timeout=timeout,
                )
                ct = resp.headers.get("Content-Type", "")
                if resp.status_code == 200:
                    # Overpass sometimes replies text/plain with JSON content; try to parse anyway
                    try:
                        return resp.json()
                    except Exception as e:
                        last_err = RuntimeError(
                            f"{endpoint} -> 200 but JSON parse failed: {e}; "
                            f"Content-Type={ct}; Snippet={resp.text[:200]!r}"
                        )
                else:
                    last_err = RuntimeError(
                        f"{endpoint} -> HTTP {resp.status_code}; "
                        f"Content-Type={ct}; Snippet={resp.text[:200]!r}"
                    )
                time.sleep(delay)
                delay *= backoff_base
            except RequestException as e:
                last_err = e
                time.sleep(delay)
                delay *= backoff_base
    raise RuntimeError(f"Overpass request failed across all endpoints. Last error: {last_err}")

# ---------------------------------------
# Build an Overpass QL to target Germany
# ---------------------------------------
# We query nodes/ways/relations inside the Germany admin area (ISO3166-1=DE, admin_level=2).
# We ask for tags + 'center' so we can turn everything into points easily.
OVERPASS_QL = r"""
[out:json][timeout:180];
area["ISO3166-1"="DE"][admin_level=2]->.de;
(
  node(area.de)["railway"~"^(station|yard|container_terminal)$"]["abandoned"!="yes"]["disused"!="yes"];
  way(area.de)["railway"~"^(station|yard|container_terminal)$"]["abandoned"!="yes"]["disused"!="yes"];
  relation(area.de)["railway"~"^(station|yard|container_terminal)$"]["abandoned"!="yes"]["disused"!="yes"];

  node(area.de)["building"="station"]["abandoned"!="yes"]["disused"!="yes"];
  way(area.de)["building"="station"]["abandoned"!="yes"]["disused"!="yes"];
  relation(area.de)["building"="station"]["abandoned"!="yes"]["disused"!="yes"];
);
out tags center;
"""

# -------------------------------
# Parse -> GeoDataFrame (points)
# -------------------------------
def _element_point(el):
    """Return shapely Point (lon, lat) for an element using node coords or 'center'."""
    if el.get("type") == "node":
        lat, lon = el.get("lat"), el.get("lon")
        if lat is None or lon is None:
            return None
        return Point(lon, lat)
    else:
        center = el.get("center") or {}
        lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            return None
        return Point(lon, lat)

def elements_to_gdf(data):
    rows = []
    for el in data.get("elements", []):
        geom = _element_point(el)
        if geom is None:
            continue
        tags = el.get("tags", {}) or {}
        rows.append({
            "osm_id": el.get("id"),
            "osm_type": el.get("type"),
            "railway": tags.get("railway"),
            "building": tags.get("building"),
            "name": tags.get("name"),
            "public_transport": tags.get("public_transport"),
            "tags": tags,
            "geometry": geom,
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # If an element matched multiple sub-queries, dedupe on (type,id)
    gdf = gdf.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    return gdf

def is_disused_abandoned(tags: dict) -> bool:
    if not tags:
        return False
    # explicit keys/values
    if tags.get("abandoned") == "yes" or tags.get("disused") == "yes":
        return True
    if tags.get("railway") in {"abandoned", "disused"}:
        return True
    # namespaced keys like "disused:railway", "abandoned:railway"
    for k in tags.keys():
        if "abandoned" in k or "disused" in k:
            return True
    return False

def main():
    # 1) Fetch from Overpass
    data = query_overpass(OVERPASS_QL)

    # 2) Convert to GeoDataFrame
    all_gdf = elements_to_gdf(data)
    print(f"Fetched {len(all_gdf)} raw elements with point geometry")

    # 3) Filter 1: drop abandoned/disused
    mask_disused = all_gdf["tags"].apply(is_disused_abandoned)
    filtered_gdf = all_gdf[~mask_disused].copy()
    print(f"Kept {len(filtered_gdf)} non-abandoned POIs out of {len(all_gdf)} total")
    print(f"Deleted POIs (abandoned/disused): {mask_disused.sum()}")

    # 4) Filter 2: drop public_transport unless explicitly 'no'
    mask_pt_drop = filtered_gdf["public_transport"].apply(lambda v: (v is not None) and (v != "no"))
    pois = filtered_gdf[~mask_pt_drop].copy()
    print(f"Kept {len(pois)} POIs out of {len(filtered_gdf)} after public_transport filter")
    print(f"Deleted POIs (public_transport present & not 'no'): {mask_pt_drop.sum()}")

    # 5) Category splits
    pois_yard = pois[pois["railway"] == "yard"]
    pois_stations = pois[pois["railway"] == "station"]
    pois_cterminals = pois[pois["railway"] == "container_terminal"]
    pois_building = pois[pois["building"] == "station"]

    # Overlaps
    common_idx_1 = pois_stations.index.intersection(pois_yard.index)
    common_idx_2 = pois_stations.index.intersection(pois_cterminals.index)
    common_idx_3 = pois_yard.index.intersection(pois_cterminals.index)
    common_idx_4 = pois_yard.index.intersection(common_idx_2)

    print(f"→ Intersection count (station ∩ yard): {len(common_idx_1)}")
    print(f"→ Intersection count (station ∩ container_terminal): {len(common_idx_2)}")
    print(f"→ Intersection count (container_terminal ∩ yard): {len(common_idx_3)}")
    print(f"→ Intersection count (container_terminal ∩ yard ∩ station): {len(common_idx_4)}")

    print("Total number of POIs:", len(pois))
    print("→ Stations:", len(pois_stations))
    print("→ Yard:", len(pois_yard))
    print("→ Container_terminal:", len(pois_cterminals))
    print("→ Building:station:", len(pois_building))

    # 6) Save to GeoPackage
    out_path = "final_pois.gpkg"
    layer_name = "rail_pois"
    # Keep the same minimal column set you used before (plus ids)
    save_cols = ["railway", "building", "name", "public_transport", "osm_type", "osm_id", "geometry"]
    pois[save_cols].to_file(out_path, layer=layer_name, driver="GPKG")
    print(f"Saved {len(pois)} POIs → {out_path} (layer '{layer_name}')")

if __name__ == "__main__":
    main()
