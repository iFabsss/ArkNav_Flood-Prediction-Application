import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# --- Load datasets once at startup ---
flood_zones = gpd.read_file('data/MetroManila 5 Years Project NOAH/MetroManila_Flood_5year.shp')
flood_history = pd.read_csv('data/AEGISDataset_NCR.csv')

# Ensure flood zones are in EPSG:3857 (meters)
if flood_zones.crs.to_epsg() != 3857:
    flood_zones = flood_zones.to_crs(epsg=3857)

# === Hazard map legend ===
hazard_map = {
    1: "Low (0–0.5m)",
    2: "Medium (>0.5–1.5m)",
    3: "High (>1.5m)"
}

# --- Function to check if coordinate is inside a flood zone ---
def find_flood_zone(lat, lon):
    # Convert to meters projection
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)

    # Check which polygons contain this point
    matching_zones = flood_zones[flood_zones.contains(point.iloc[0])]

    if matching_zones.empty:
        # If not inside, compute distance to nearest polygon boundary
        distances = flood_zones.distance(point.iloc[0])
        nearest_distance = distances.min()

        return {
            "in_zone": False,
            "hazard_code": 0,
            "hazard_level": "Outside zone",
            "proximity_m": float(nearest_distance)
        }


    # Get hazard code (Var column in shapefile)
    hazard_code = int(matching_zones.iloc[0]['Var'])
    hazard_description = hazard_map.get(hazard_code, "Unknown")

    return {
        "in_zone": True,
        "hazard_code": hazard_code,
        "hazard_level": hazard_description,
        "proximity_m": 0.0 # Inside zone
    }

# --- Apply to AEGIS dataset ---
results = []
for _, row in flood_history.iterrows():
    zone_info = find_flood_zone(row["lat"], row["lon"])
    results.append({
        "lat": row["lat"],
        "lon": row["lon"],
        "flood_height": row["flood_height"],
        "elevation": row["elevation"],
        "precipitation": row["precipitation"],
        "in_zone": zone_info["in_zone"],
        "hazard_code": zone_info["hazard_code"],
        "hazard_level": zone_info["hazard_level"],
        "proximity_m": zone_info["proximity_m"]
    })

# --- Create merged DataFrame ---
merged = pd.DataFrame(results)

# Save merged dataset
merged.to_csv("data/AEGISDataset_NCR_WithHazard.csv", index=False)
print("✅ Successfully merged flood zones with AEGIS dataset (using meter-based projection).")
