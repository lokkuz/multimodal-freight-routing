#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_add_weights_template.py

Add weight columns (time, cost, emissions) to a merged multimodal network.

Inputs (GPKG):
  - layer 'network_edges': [u, v, mode in {'road','rail','connection'}, geometry], CRS=EPSG:25832
  - layer 'network_nodes': [node_id, geometry], CRS=EPSG:25832

Outputs:
  - Writes GPKG with same layers; 'network_edges' gains:
      length_m, time_min, cost_eur, co2e_kg,
      weight_time, weight_cost, weight_emission

Logic:
  length_km = length_m / 1000

  For mode in {'road','rail'}:
    time_min  = (length_km / speed_kmh[mode]) * 60
    cost_eur  =  length_km * cost_eur_per_km[mode]
    co2e_kg   =  length_km * emis_kg_per_km[mode]

  For mode == 'connection':
    time_min  = conn_fixed_time_min             # constant (no length dependency)
    cost_eur  = conn_fixed_cost_eur             # constant
    co2e_kg   = conn_fixed_emis_kg              # constant

Your provided defaults are used; all are overridable via CLI.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import geopandas as gpd

# ---------------- Paths / layers ----------------
BASE = Path(".").resolve()
IN_GPKG  = BASE / "graphs" / "multimodal_network.gpkg"
OUT_GPKG = BASE / "graphs" / "multimodal_network_with_weights.gpkg"
EDGES_LAYER = "network_edges"
NODES_LAYER = "network_nodes"
EPSG_METRIC = 25832  # meters

# ---------------- Defaults (from your spec) ----------------
DEFAULT_TEU = 80
DEFAULT_WEIGHT_PER_TEU = 11.5  # t per TEU

# Speeds (km/h)
DEFAULT_SPEED_ROAD = 80.0
DEFAULT_SPEED_RAIL = 55.0
# connection speed not used (fixed), kept as 0.0 for clarity

# Cost (€/km); these are TOTAL €/km for the whole shipment,
# already multiplied by TEU in the defaults below.
def _default_costkm_road(teu: int) -> float: return 1.145 * teu
def _default_costkm_rail(teu: int) -> float: return 0.05 * teu

# Connection fixed cost (€/TEU) -> total fixed cost
def _default_conn_fixed_cost(teu: int) -> float: return 50.0 * teu

# Emissions (kg CO2e per km) for the whole shipment
def _default_emisk_road(teu: int, w_per_teu: float) -> float:
    # 0.1295 kg CO2e per t-km * (t/TEU) * TEU
    return 0.1295 * w_per_teu * teu

def _default_emisk_rail(teu: int, w_per_teu: float) -> float:
    return 0.024 * w_per_teu * teu

# Connection fixed time/cost/emissions
def _default_conn_fixed_time_min(teu: int) -> float:  # minutes
    return 111.0 + 0.028 * 60 * teu

DEFAULT_CONN_FIXED_EMIS_KG = 0.0  # fixed

# ---------------- IO ----------------
def load_layers(gpkg: Path):
    if not gpkg.exists():
        print(f"ERROR: input GPKG not found: {gpkg}", file=sys.stderr)
        sys.exit(1)
    edges = gpd.read_file(gpkg, layer=EDGES_LAYER)
    nodes = gpd.read_file(gpkg, layer=NODES_LAYER)
    if edges.crs is None: edges = edges.set_crs(epsg=EPSG_METRIC, allow_override=True)
    if nodes.crs is None: nodes = nodes.set_crs(epsg=EPSG_METRIC, allow_override=True)
    if edges.crs.to_epsg() != EPSG_METRIC: edges = edges.to_crs(EPSG_METRIC)
    if nodes.crs.to_epsg() != EPSG_METRIC: nodes = nodes.to_crs(EPSG_METRIC)
    edges["mode"] = edges["mode"].astype(str).str.lower()
    edges["u"] = edges["u"].astype(int); edges["v"] = edges["v"].astype(int)
    nodes["node_id"] = nodes["node_id"].astype(int)
    return edges, nodes

def ensure_length_m(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "length_m" not in edges.columns:
        edges["length_m"] = edges.geometry.length.astype(float)
    else:
        mask = edges["length_m"].isna() | (edges["length_m"] <= 0)
        if mask.any():
            edges.loc[mask, "length_m"] = edges.loc[mask].geometry.length.astype(float)
    return edges

# ---------------- Core weighting ----------------
def apply_weights(
    edges: gpd.GeoDataFrame,
    teu: int,
    weight_per_teu: float,
    speed_road: float,
    speed_rail: float,
    costkm_road: float,
    costkm_rail: float,
    conn_fixed_time_min: float,
    conn_fixed_cost_eur: float,
    conn_fixed_emis_kg: float,
    emis_road_per_km: float,
    emis_rail_per_km: float,
    road_speed_secondary_kmh: float = 60.0,
    road_secondary_tags: tuple[str, ...] = ("secondary", "secondary_link"),
):
    """
    Adds time_min, cost_eur, co2e_kg and weight_* columns.
    - Road/Rail: length-dependent.
    - Connection: fixed values.
    - Roads with highway in road_secondary_tags use road_speed_secondary_kmh.
    """
    e = ensure_length_m(edges.copy())
    e["mode"] = e["mode"].astype(str).str.lower()
    length_km = e["length_m"].astype(float) / 1000.0

    # Masks
    m_road = e["mode"] == "road"
    m_rail = e["mode"] == "rail"
    m_conn = e["mode"] == "connection"

    # ---------- TIME (minutes) ----------
    # Build per-row speed array only for road/rail (connection is fixed)
    spd_kmh = np.zeros(len(e), dtype=float)

    # Base speeds
    spd_kmh[m_rail] = float(speed_rail)
    spd_kmh[m_road] = float(speed_road)

    # Optional downgrade for specific road classes if 'highway' exists
    if "highway" in e.columns:
        # Normalize to lowercase strings; handle lists or semicolon-separated tags
        def norm_hw(x):
            if x is None:
                return ""
            s = str(x).lower()
            # If it's like "secondary;residential", split and match any
            if ";" in s:
                return [p.strip() for p in s.split(";")]
            return s

        hw_series = e["highway"].apply(norm_hw)

        # Build boolean mask where the highway type matches any of the secondary tags
        sec_tags = set(t.strip().lower() for t in road_secondary_tags)
        if isinstance(hw_series.iloc[0], list):
            m_secondary = hw_series.apply(lambda lst: any(t in sec_tags for t in lst)).to_numpy(bool)
        else:
            m_secondary = hw_series.isin(sec_tags).to_numpy(bool)

        # Apply only on road edges
        m_secondary = m_secondary & m_road.to_numpy()
        spd_kmh[m_secondary] = float(road_speed_secondary_kmh)

    # Guard against zero
    spd_kmh = np.where(spd_kmh <= 0, 1e-6, spd_kmh)

    time_min = np.zeros(len(e), dtype=float)
    time_min[m_road | m_rail] = (length_km[m_road | m_rail] / spd_kmh[m_road | m_rail]) * 60.0
    time_min[m_conn] = float(conn_fixed_time_min)  # fixed for connections

    # ---------- COST (EUR) ----------
    cost_eur = np.zeros(len(e), dtype=float)
    cost_eur[m_road] = length_km[m_road] * float(costkm_road)
    cost_eur[m_rail] = length_km[m_rail] * float(costkm_rail)
    cost_eur[m_conn] = float(conn_fixed_cost_eur)  # fixed

    # ---------- EMISSIONS (kg CO2e) ----------
    co2e_kg = np.zeros(len(e), dtype=float)
    co2e_kg[m_road] = length_km[m_road] * float(emis_road_per_km)
    co2e_kg[m_rail] = length_km[m_rail] * float(emis_rail_per_km)
    co2e_kg[m_conn] = float(conn_fixed_emis_kg)  # fixed

    # Attach
    e["time_min"] = time_min
    e["cost_eur"] = cost_eur
    e["co2e_kg"]  = co2e_kg

    # Routing weight aliases
    e["weight_time"]     = e["time_min"]
    e["weight_cost"]     = e["cost_eur"]
    e["weight_emission"] = e["co2e_kg"]

    return e


def write_output(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, out_path: Path, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        out_path.unlink()
    edges.to_file(out_path, layer=EDGES_LAYER, driver="GPKG")
    nodes.to_file(out_path, layer=NODES_LAYER, driver="GPKG")
    print(f"✅ Wrote: {out_path}")
    # Quick report
    by_mode = (edges.groupby("mode")[["length_m","time_min","cost_eur","co2e_kg"]]
                    .sum()
                    .rename(columns={"length_m":"len_m_total"}))
    print("\n=== Mode totals ===")
    print((by_mode.assign(len_km_total = by_mode["len_m_total"]/1000.0)
                 [["len_km_total","time_min","cost_eur","co2e_kg"]]
                 .round(2)))
    print("===================\n")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Add time/cost/CO2e weights to multimodal network (length-based for road/rail; fixed for connection).")
    ap.add_argument("--in",  dest="in_gpkg",  type=str, default=str(IN_GPKG), help="Input GPKG")
    ap.add_argument("--out", dest="out_gpkg", type=str, default=str(OUT_GPKG), help="Output GPKG")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")

    # Shipment
    ap.add_argument("--teu", type=int, default=DEFAULT_TEU, help="Number of TEU for totals")
    ap.add_argument("--weight-per-teu", type=float, default=DEFAULT_WEIGHT_PER_TEU, help="t/TEU for emissions calc")

    # Speeds (km/h)
    ap.add_argument("--speed-road", type=float, default=DEFAULT_SPEED_ROAD)
    ap.add_argument("--speed-rail", type=float, default=DEFAULT_SPEED_RAIL)
    ap.add_argument("--road-speed-secondary-kmh", type=float, default=60.0,
                    help="Speed to use for road edges with highway in the secondary set (km/h)")
    ap.add_argument("--road-secondary-tags", type=str, default="secondary,secondary_link",
                    help="Comma-separated highway tags that should use the secondary speed")

    # Cost (€/km) totals for shipment (defaults derived from TEU)
    ap.add_argument("--costkm-road", type=float, default=None,
                    help="€ per km for ROAD (total for shipment). Default = 0.05 * TEU")
    ap.add_argument("--costkm-rail", type=float, default=None,
                    help="€ per km for RAIL (total for shipment). Default = 1.145 * TEU")

    # Connection fixed penalties
    ap.add_argument("--conn-fixed-time-min", type=float, default=None,
                    help="Fixed minutes for CONNECTION edges (default = 60 + 0.028*TEU)")
    ap.add_argument("--conn-fixed-cost-eur", type=float, default=None,
                    help="Fixed € for CONNECTION edges (default = 50 * TEU)")
    ap.add_argument("--conn-fixed-emis-kg", type=float, default=DEFAULT_CONN_FIXED_EMIS_KG,
                    help="Fixed kg CO2e for CONNECTION edges")

    # Emissions (kg CO2e per km), totals for shipment (defaults derived from TEU and weight/TEU)
    ap.add_argument("--emis-road-per-km", type=float, default=None,
                    help="kg CO2e per km for ROAD (total shipment). Default = 0.1295 * weight_per_TEU * TEU")
    ap.add_argument("--emis-rail-per-km", type=float, default=None,
                    help="kg CO2e per km for RAIL (total shipment). Default = 0.024 * weight_per_TEU * TEU")

    args = ap.parse_args()

    # Resolve TEU-dependent defaults
    teu = int(args.teu)
    wpt = float(args.weight_per_teu)

    costkm_road = args.costkm_road if args.costkm_road is not None else _default_costkm_road(teu)
    costkm_rail = args.costkm_rail if args.costkm_rail is not None else _default_costkm_rail(teu)

    conn_fixed_time_min = args.conn_fixed_time_min if args.conn_fixed_time_min is not None else _default_conn_fixed_time_min(teu)
    conn_fixed_cost_eur = args.conn_fixed_cost_eur if args.conn_fixed_cost_eur is not None else _default_conn_fixed_cost(teu)

    emis_road_per_km = args.emis_road_per_km if args.emis_road_per_km is not None else _default_emisk_road(teu, wpt)
    emis_rail_per_km = args.emis_rail_per_km if args.emis_rail_per_km is not None else _default_emisk_rail(teu, wpt)

    # Load & compute
    edges, nodes = load_layers(Path(args.in_gpkg))
    edges_w = apply_weights(
        edges,
        teu=teu,
        weight_per_teu=wpt,
        speed_road=float(args.speed_road),
        speed_rail=float(args.speed_rail),
        costkm_road=float(costkm_road),
        costkm_rail=float(costkm_rail),
        conn_fixed_time_min=float(conn_fixed_time_min),
        conn_fixed_cost_eur=float(conn_fixed_cost_eur),
        conn_fixed_emis_kg=float(args.conn_fixed_emis_kg),
        emis_road_per_km=float(emis_road_per_km),
        emis_rail_per_km=float(emis_rail_per_km),
        road_speed_secondary_kmh=float(args.road_speed_secondary_kmh),
        road_secondary_tags=tuple(t.strip() for t in args.road_secondary_tags.split(",")),
    )

    write_output(edges_w, nodes, Path(args.out_gpkg), overwrite=args.overwrite)

if __name__ == "__main__":
    main()
