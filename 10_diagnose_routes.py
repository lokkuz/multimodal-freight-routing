#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_diagnose_routes.py

Diagnose optimized multimodal route outputs by summarizing, PER MODE:
- distance (km)
- time (hr)
- cost (€)
- emission (kg CO2e)
- segment count

Works on your per-weight route GeoJSONs (and GPKG). Auto-discovers files from
--orig/--dest or accepts explicit --files.

Outputs:
- prints a clean table per file
- optional CSV with all summaries merged
"""

from pathlib import Path
import argparse
import re
import sys
import warnings

import pandas as pd
import geopandas as gpd

BASE = Path(".").resolve()
DEF_ROUTES_DIR = BASE / "graphs" / "routes"
DEF_OUT_CSV    = BASE / "graphs" / "routes" / "diagnostics_summary_C-B.csv"

KNOWN_WEIGHTS = {"distance", "time", "cost", "emission"}

def _parse_weight_from_name(p: Path) -> str:
    m = re.search(r'_(distance|time|cost|emission)\.(?:geojson|gpkg|geopackage)$', p.name, re.I)
    return m.group(1).lower() if m else "unknown"

def _ensure_metrics_cols(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure metrics exist, falling back to your weight columns if needed:
      distance: length_m (meters)
      time:     time_min or weight_time
      cost:     cost_eur  or weight_cost
      emission: co2e_kg   or weight_emission
    """
    if "length_m" not in gdf.columns:
        gdf["length_m"] = gdf.geometry.length.astype(float)

    def _ensure(dst, pri, alt=None, default=0.0):
        if dst in gdf.columns:
            return
        if pri in gdf.columns:
            gdf[dst] = gdf[pri]
        elif alt and alt in gdf.columns:
            gdf[dst] = gdf[alt]
        else:
            gdf[dst] = default

    _ensure("time_min", "time_min", "weight_time", 0.0)
    _ensure("cost_eur", "cost_eur", "weight_cost", 0.0)
    _ensure("co2e_kg",  "co2e_kg",  "weight_emission", 0.0)

    if "mode" in gdf.columns:
        gdf["mode"] = gdf["mode"].astype(str).str.lower()
    else:
        gdf["mode"] = "unknown"

    return gdf

def _load_route(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".gpkg", ".geopackage"}:
        try:
            g = gpd.read_file(path, layer="network_edges")
        except Exception:
            g = gpd.read_file(path)
    else:
        g = gpd.read_file(path)
    if g.crs is None:
        warnings.warn(f"{path.name}: missing CRS; assuming metric CRS (lengths in meters may be wrong).")
    return _ensure_metrics_cols(g)

def _summarize_one(path: Path, label: str) -> pd.DataFrame:
    g = _load_route(path)
    if g.empty:
        return pd.DataFrame(columns=[
            "route_label","mode","segments","distance_km","time_hr","cost_eur","co2e_kg"
        ])

    by = (g.groupby("mode", dropna=False)
            .agg(segments=("geometry", "size"),
                 distance_km=("length_m", lambda x: float(x.sum())/1000.0),
                 time_hr=("time_min", lambda x: x.sum() / 60.0),
                 cost_eur=("cost_eur", "sum"),
                 co2e_kg=("co2e_kg", "sum"))
            .reset_index())

    totals = pd.DataFrame([{
        "mode": "ALL",
        "segments": int(by["segments"].sum()),
        "distance_km": float(g["length_m"].sum())/1000.0,
        "time_hr": float(g["time_min"].sum())/60.0,
        "cost_eur": float(g["cost_eur"].sum()),
        "co2e_kg": float(g["co2e_kg"].sum()),
    }])

    out = pd.concat([by, totals], ignore_index=True)
    out.insert(0, "route_label", label)
    return out

def discover_files(orig: str, dest: str, routes_dir: Path, weights: list[str]) -> list[Path]:
    def _slug(s: str) -> str:
        return re.sub(r'[^A-Za-z0-9]+', '-', s.strip().lower()).strip("-")
    o, d = _slug(orig), _slug(dest)
    return [routes_dir / f"multimodal_{o}_to_{d}_{w}.geojson" for w in weights]

def print_table(df: pd.DataFrame):
    if df.empty:
        print("(no data)")
        return
    cols = ["route_label","mode","segments","distance_km","time_hr","cost_eur","co2e_kg"]
    df = df[cols].copy()
    def fmtf(x): return f"{x:,.2f}"
    df["distance_km"] = df["distance_km"].map(fmtf)
    df["time_hr"]     = df["time_hr"].map(fmtf)
    df["cost_eur"]    = df["cost_eur"].map(fmtf)
    df["co2e_kg"]     = df["co2e_kg"].map(fmtf)
    print(df.to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description="Diagnose multimodal routes: per-mode distance (km), time (hr), cost (€), emission (kg).")
    ap.add_argument("--routes-dir", type=str, default=str(DEF_ROUTES_DIR),
                    help="Directory where your route GeoJSONs live")
    ap.add_argument("--files", type=str, nargs="*", default=[],
                    help="Explicit list of route files to analyze (GeoJSON/GPKG)")
    ap.add_argument("--orig", type=str, default="Berlin",
                    help="Origin name (used only if --files not given)")
    ap.add_argument("--dest", type=str, default="Cottbus",
                    help="Destination name (used only if --files not given)")
    ap.add_argument("--weights", type=str, default="distance,time,cost,emission",
                    help="Comma list used for auto-discovery (if --files not given)")
    ap.add_argument("--out-csv", type=str, default=str(DEF_OUT_CSV),
                    help="Write merged summary CSV here (set empty to skip)")
    args = ap.parse_args()

    files = [Path(f) for f in args.files] if args.files else \
            discover_files(args.orig, args.dest, Path(args.routes_dir),
                           [w.strip().lower() for w in args.weights.split(",") if w.strip()])

    if not files:
        print("No files to analyze. Provide --files or --orig/--dest.", file=sys.stderr)
        sys.exit(2)

    all_rows = []
    for fp in files:
        label = _parse_weight_from_name(fp)
        try:
            df = _summarize_one(fp, label)
        except FileNotFoundError:
            print(f"⚠️  Missing: {fp}")
            continue
        if df.empty:
            print(f"⚠️  Empty route: {fp}")
            continue

        print(f"\n=== {fp.name} ({label}) ===")
        print_table(df)
        all_rows.append(df)

        # Consistency check if weight_used is present
        try:
            g = _load_route(fp)
            if "weight_used" in g.columns and label in KNOWN_WEIGHTS:
                total_used = float(g["weight_used"].sum())
                if label == "distance":
                    ref = float(g["length_m"].sum());         unit_ref, unit_used = "m", "m"
                elif label == "time":
                    ref = float(g["time_min"].sum());         unit_ref, unit_used = "min", "min"
                elif label == "cost":
                    ref = float(g["cost_eur"].sum());         unit_ref, unit_used = "€", "€"
                else:
                    ref = float(g["co2e_kg"].sum());          unit_ref, unit_used = "kg", "kg"
                diff = total_used - ref
                print(f"• Check: sum(weight_used) = {total_used:,.3f} {unit_used}; reference sum = {ref:,.3f} {unit_ref}; Δ = {diff:,.3f}")
        except Exception:
            pass

    if not all_rows:
        print("\nNo summaries produced (files missing/empty).")
        sys.exit(1)

    merged = pd.concat(all_rows, ignore_index=True)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)
        print(f"\n🧾 Wrote merged diagnostics CSV: {out_csv}")

if __name__ == "__main__":
    main()
