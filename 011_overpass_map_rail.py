import requests
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path

import matplotlib.pyplot as plt
import pyproj
from shapely.ops import transform
import os
import json
import datetime


# Inputs:
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "input_files" / "derived"
OUTPUT_FILE = OUTPUT_DIR / "germany_rail_edges.geojson"


##############################
# Auxiliar Functions
##############################

# Query railways in Germany
# ----------------------------
def query_railways(kinds=("rail",), include_service=False):
    kinds_re = "|".join(kinds)
    # Exclude yard/siding/spur/etc unless you explicitly want them
    service_filter = "" if include_service else '["service"!~"."]'
    rail_query = f"""
    [out:json][timeout:1800];
    area["ISO3166-1"="DE"]->.searchArea;
    (
      way["railway"~"^({kinds_re})$"]{service_filter}(area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": rail_query})
    r.raise_for_status()
    return r.json()


def parse_rail_data(data):
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in data["elements"] if e["type"] == "node"}
    features = []
    for e in data["elements"]:
        if e["type"] == "way" and "nodes" in e:
            coords = [nodes[n] for n in e["nodes"] if n in nodes]
            if len(coords) >= 2:
                tags = e.get("tags", {})
                features.append({
                    "geometry": LineString(coords),
                    "railway": tags.get("railway"),     # rail, light_rail, subway, tram, …
                    "ref": tags.get("ref"),
                    "name": tags.get("name")
                })
    return features



# Query Germany Border
# ----------------------------
def query_germany_border():
    border_query = """
    [out:json];
    rel["name"="Deutschland"]["admin_level"="2"];
    out body;
    >;
    out skel qt;
    """
    response = requests.post("https://overpass-api.de/api/interpreter", data={"data": border_query})
    return response.json()


def parse_border_data(data):
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in data["elements"] if e["type"] == "node"}
    border_lines = []

    for way in data["elements"]:
        if way["type"] == "way" and "nodes" in way and len(way["nodes"]) >= 2:
            coords = [nodes[n] for n in way["nodes"] if n in nodes]
            if len(coords) >= 2:
                border_lines.append({"geometry": LineString(coords)})

    return gpd.GeoDataFrame(border_lines, crs="EPSG:4326")



# Save edges to JSON with length in meters
# ----------------------------
def save_rail_edges_to_json(features, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Use geodesic length to avoid UTM zone issues across Germany
    geod = pyproj.Geod(ellps="WGS84")
    edges_data = []
    for f in features:
        length_m = geod.geometry_length(f["geometry"])
        edges_data.append({
            "geometry": list(f["geometry"].coords),  # [lon, lat]
            "railway": f["railway"],
            "ref": f.get("ref"),
            "name": f.get("name"),
            "length": length_m
        })
    with open(filename, "w") as fh:
        json.dump(edges_data, fh, indent=4)



# Plot Map functions
# ----------------------------
def plot_map(gdf_border, gdf_rail, output_path="germany_rail_map.png"):
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf_border.plot(ax=ax, edgecolor="black", linewidth=2, label="German Border")
    gdf_rail.plot(ax=ax, color="green", linewidth=0.7, label="Rail")
    plt.title("Germany Rail Network")
    plt.axis("off")
    plt.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


##############################
# Main Workflow
##############################
if __name__ == "__main__":
    # 1) Fetch and parse rails
    rail_data = query_railways(kinds=("rail",), include_service=False)
    rail_features = parse_rail_data(rail_data)
    gdf_rail = gpd.GeoDataFrame(rail_features, crs="EPSG:4326")

    # 2) Border
    border_data = query_germany_border()
    gdf_border = parse_border_data(border_data)

    # 3) Save JSON
    save_rail_edges_to_json(rail_features, OUTPUT_FILE)

    # 4) Plot
    plot_map(gdf_border, gdf_rail)
