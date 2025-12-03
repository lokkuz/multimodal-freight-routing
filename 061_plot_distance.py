#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ONLY the route GeoJSON, coloring segments by 'mode' (road/rail/connection).
Defaults:
  ROUTE_F = graphs/hh_muc_route_edges.geojson
  OUT_PNG = graphs/hh_muc_route_by_mode.png
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# Defaults
BASE_DIR = Path(".").resolve()
ROUTE_F = BASE_DIR / "graphs" / "hh_muc_route.geojson"
OUT_PNG = BASE_DIR / "graphs" / "hh_muc_route_by_mode.png"

# Plot in Web Mercator
PLOT_EPSG = 3857

def main():
    if not ROUTE_F.exists():
        raise FileNotFoundError(f"Route file not found: {ROUTE_F}")

    # Load
    route = gpd.read_file(ROUTE_F)
    if route.crs is None:
        route = route.set_crs(epsg=4326, allow_override=True)
    route = route.to_crs(epsg=PLOT_EPSG)

    # Ensure a 'mode' column (fallback to 'unknown')
    if "mode" not in route.columns:
        route = route.copy()
        route["mode"] = "unknown"

    # Normalize values
    def norm(x: str) -> str:
        x = (x or "").lower()
        if "rail" in x: return "rail"
        if "road" in x or "highway" in x: return "road"
        if "conn" in x or "link" in x: return "connection"
        return "unknown"
    route["mode"] = route["mode"].astype(str).map(norm)

    # Styles
    styles = {
        "road":       dict(color="gray",   linewidth=2.0, alpha=0.9, zorder=2),
        "rail":       dict(color="blue",   linewidth=2.6, alpha=0.9, zorder=3, linestyle="--"),
        "connection": dict(color="orange", linewidth=3.0, alpha=1.0, zorder=4),
        "unknown":    dict(color="lightgray", linewidth=1.5, alpha=0.6, zorder=1),
    }

    fig, ax = plt.subplots(figsize=(9, 10))

    # Plot by mode
    draw_order = ["road", "rail", "connection", "unknown"]
    for m in draw_order:
        g = route[route["mode"] == m]
        if len(g) > 0:
            g.plot(ax=ax, label=m, **styles[m])

    # Start/End markers from first/last segment
    try:
        first_ls = route.geometry.iloc[0]
        last_ls  = route.geometry.iloc[-1]
        x0, y0 = list(first_ls.coords)[0]
        x1, y1 = list(last_ls.coords)[-1]
        ax.scatter([x0], [y0], s=40, color="green", zorder=10, label="origin")
        ax.scatter([x1], [y1], s=60, color="red", marker="X", zorder=11, label="destination")
    except Exception:
        pass

    # Tight bounds around route
    minx, miny, maxx, maxy = route.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_axis_off()
    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="lower left", frameon=False)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Route-only plot saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
