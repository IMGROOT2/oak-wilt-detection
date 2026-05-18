"""
Generate a presentation-quality PNG showing a single Texas A&M oak wilt polygon
overlaid on satellite imagery.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import mapping
import numpy as np

# Load the TAMU shapefile
gdf = gpd.read_file("data/OW_2020_2026/OW_2020_2026.shp")

# Hand-picked: idx=268, ~9 acres in Gillespie County
# Clear browning/die-off visible against healthy green canopy
example = gdf.loc[[268]].copy()

print(f"Selected polygon: {example.iloc[0]['County']} County")
print(f"  Acres: {example.iloc[0]['Acres']:.1f}")
print(f"  Diagnosis: {example.iloc[0]['Diagnosis']}")

# Reproject to Web Mercator for satellite tile overlay
example_wm = example.to_crs(epsg=3857)

# Get bounds and add padding (30% on each side for context)
bounds = example_wm.total_bounds  # minx, miny, maxx, maxy
dx = bounds[2] - bounds[0]
dy = bounds[3] - bounds[1]
pad = max(dx, dy) * 0.6  # generous padding so you see surrounding trees
padded_bounds = [
    bounds[0] - pad,
    bounds[1] - pad,
    bounds[2] + pad,
    bounds[3] + pad,
]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)

# Plot the polygon with a bold outline, semi-transparent fill
example_wm.plot(
    ax=ax,
    facecolor="#FF4444",
    edgecolor="#FFFFFF",
    linewidth=2.5,
    alpha=0.35,
    zorder=3,
)

# Also draw a stronger outline on top
example_wm.boundary.plot(
    ax=ax,
    color="#FF2222",
    linewidth=2.5,
    zorder=4,
)

# Set extent to padded bounds
ax.set_xlim(padded_bounds[0], padded_bounds[2])
ax.set_ylim(padded_bounds[1], padded_bounds[3])

# Add satellite basemap (Esri World Imagery)
ctx.add_basemap(
    ax,
    source=ctx.providers.Esri.WorldImagery,
    zoom="auto",
    attribution=False,
)

# Clean up axes for presentation
ax.set_axis_off()

# Add a subtle title
# Texas FIPS code to county name lookup for common codes
FIPS_TO_COUNTY = {
    "029": "Bexar", "019": "Bandera", "053": "Burnet", "091": "Comal",
    "099": "Coryell", "171": "Gillespie", "187": "Guadalupe",
    "209": "Hays", "259": "Kendall", "265": "Kerr", "267": "Kimble",
    "299": "Llano", "319": "Mason", "325": "Medina", "333": "Mills",
    "453": "Travis", "491": "Williamson", "027": "Bell", "031": "Blanco",
    "035": "Bosque", "049": "Brown", "193": "Hamilton",
}
fips = example.iloc[0]["County"]
county = FIPS_TO_COUNTY.get(fips, f"FIPS {fips}")
acres = example.iloc[0]["Acres"]
ax.set_title(
    f"Oak Wilt Detection Polygon — {county} County ({acres:.1f} acres)",
    fontsize=14,
    fontweight="bold",
    color="#FFFFFF",
    pad=12,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000", alpha=0.7),
)

plt.tight_layout()
plt.savefig(
    "tamu_polygon_satellite.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="black",
    pad_inches=0.3,
)
plt.close()
print("\nSaved: tamu_polygon_satellite.png")
