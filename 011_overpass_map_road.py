import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
import os
import json
from shapely.geometry import LineString
from pathlib import Path
from shapely.ops import transform

# Inputs:
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "input_files" / "derived"
OUTPUT_FILE = OUTPUT_DIR / "germany_all_road_edges.geojson"


##############################
# 1) Query road network in Germany (motorways, trunk, primary and secondary roads)
##############################
def query_highways():
    road_query = """
    [out:json][timeout:3600];
    area["name"="Deutschland"]->.searchArea;
    (
      way["highway"="motorway"](area.searchArea);
      way["highway"="motorway_link"](area.searchArea);
      way["highway"="trunk"](area.searchArea);
      way["highway"="trunk_link"](area.searchArea);
      way["highway"="primary"](area.searchArea);
      way["highway"="primary_link"](area.searchArea);
       way["highway"="secondary"](area.searchArea);
      way["highway"="secondary_link"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.post("https://overpass-api.de/api/interpreter", data={"data": road_query})
    return response.json()


def parse_road_data(data):
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in data["elements"] if e["type"] == "node"}
    features = []

    for e in data["elements"]:
        if e["type"] == "way":
            coords = [nodes[n] for n in e["nodes"] if n in nodes]
            if len(coords) >= 2:
                tags = e.get("tags", {})
                features.append({
                    "geometry": LineString(coords),
                    "highway": tags.get("highway"),
                    "ref": tags.get("ref"),
                    "name": tags.get("name")
                })
    return features


##############################
# 2) Query German border
##############################
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


##############################
# 3) Save edges to JSON with length in meters
##############################
def save_edges_to_json(features, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True).transform
    edges_data = []

    for feature in features:
        geometry = transform(project, feature["geometry"])
        length = geometry.length
        edges_data.append({
            "geometry": list(feature["geometry"].coords),
            "highway": feature["highway"],
            "ref": feature.get("ref"),
            "name": feature.get("name"),
            "length": length
        })

    with open(filename, "w") as f:
        json.dump(edges_data, f, indent=4)


##############################
# 4) Plot Map
##############################
def plot_map(gdf_border, gdf_autobahn, output_path="germany_overpass_map.png"):
    fig, ax = plt.subplots(figsize=(12, 12))

    gdf_border.plot(ax=ax, edgecolor="black", linewidth=2, label="German Border")
    gdf_autobahn.plot(ax=ax, color="red", linewidth=1, label="Autobahn (motorway + link)")

    plt.title("German Road Network: motorways, trunks, primary and secondary roads")
    plt.axis("off")
    plt.legend()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


##############################
# Main
##############################
if __name__ == "__main__":
    # Query and process highway data
    road_data = query_highways()
    highway_features = parse_road_data(road_data)

    # Filter Autobahns
    autobahn_features = [f for f in highway_features if f["highway"] in ["motorway", "motorway_link","trunk","trunk_link",
                                                                         "primary","primary_link","secondary","secondary_link"]]
    gdf_autobahn = gpd.GeoDataFrame(autobahn_features, crs="EPSG:4326")

    # Query and process border
    border_data = query_germany_border()
    gdf_border = parse_border_data(border_data)

    # Save to JSON with lengths
    save_edges_to_json(autobahn_features, OUTPUT_FILE)

    # Plot the result
    plot_map(gdf_border, gdf_autobahn)
