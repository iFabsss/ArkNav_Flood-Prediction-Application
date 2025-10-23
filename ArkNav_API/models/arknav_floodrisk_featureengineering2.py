"""
calculate_nearest_body_of_water.py

Description:
    Calculates, for each coordinate in your dataset, the nearest river/body of water
    and stores the following:
        - Distance to nearest body of water (meters)
        - Nearest river name
        - Coordinates of the nearest point on the river geometry (in degrees)

Input:
    - data\AEGISDataset_NCR_WithHazard_HazardLevelProximity.csv
    - D:\Downloads\phl_rivl_250k_namria\phl_rivl_250k_NAMRIA.shp

Output:
    - data\ArkNav_FloodRisk_Dataset.csv
"""

import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points

# === 1. LOAD DATA ===========================================================

print("📂 Loading AEGIS dataset...")
data_path = r"data\AEGISDataset_NCR_WithHazard_HazardLevelProximity.csv"
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df)} rows.")

print("\n🌊 Loading NAMRIA river shapefile...")
rivers = gpd.read_file(r"D:\Downloads\phl_rivl_250k_namria\phl_rivl_250k_NAMRIA.shp")
rivers = rivers[['RIVER_NAME', 'geometry']]

# === 2. COORDINATE SYSTEMS =================================================

# Ensure both use WGS84 initially (EPSG:4326)
if rivers.crs.to_epsg() != 4326:
    rivers = rivers.to_crs(epsg=4326)
    print("✅ Converted rivers to EPSG:4326 (WGS84).")

# Create GeoDataFrame for points
points_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['lon'], df['lat']),
    crs="EPSG:4326"
)

# Convert both to UTM zone 51N (meters)
rivers = rivers.to_crs(epsg=32651)
points_gdf = points_gdf.to_crs(epsg=32651)
print("🧭 Converted both layers to EPSG:32651 for meter-based distance computation.")

# === 3. NEAREST RIVER / WATER CALCULATION ==================================

print("\n📏 Calculating nearest body of water, name, and coordinates...")

# Build spatial index for efficiency
rivers_sindex = rivers.sindex

nearest_river_names = []
nearest_river_x = []
nearest_river_y = []
nearest_distances = []

for idx, point in enumerate(points_gdf.geometry):
    # Find possible matches within 5km bounding box
    possible_matches_index = list(rivers_sindex.intersection(point.buffer(5000).bounds))
    possible_matches = rivers.iloc[possible_matches_index]

    if possible_matches.empty:
        nearest_river_names.append("None")
        nearest_river_x.append(None)
        nearest_river_y.append(None)
        nearest_distances.append(None)
        continue

    # Find actual nearest geometry among candidates
    nearest_geom = possible_matches.distance(point).sort_values().index[0]
    nearest_river = rivers.loc[nearest_geom, 'geometry']
    river_name = rivers.loc[nearest_geom, 'RIVER_NAME']

    # Get nearest point on that river geometry
    p1, p2 = nearest_points(point, nearest_river)

    # Convert nearest river coordinate (p2) to WGS84 degrees
    p2_wgs84 = gpd.GeoSeries([p2], crs="EPSG:32651").to_crs(epsg=4326).iloc[0]

    # Store values
    nearest_river_names.append(river_name if pd.notna(river_name) else "Unnamed")
    nearest_river_x.append(p2_wgs84.x)   # longitude in degrees
    nearest_river_y.append(p2_wgs84.y)   # latitude in degrees
    nearest_distances.append(point.distance(p2))  # meters

    if (idx + 1) % 200 == 0:
        print(f"   Processed {idx + 1} / {len(points_gdf)} points...")

print("✅ Nearest river computation complete.")

# === 4. SAVE RESULTS ========================================================

points_gdf['nearest_river_name'] = nearest_river_names
points_gdf['nearest_river_lon'] = nearest_river_x
points_gdf['nearest_river_lat'] = nearest_river_y
points_gdf['dist_to_nearest_water_m'] = nearest_distances

# Drop geometry column before saving for clean CSV
points_gdf.drop(columns=['geometry'], inplace=True)

output_path = r"data\ArkNav_FloodRisk_Dataset.csv"
points_gdf.to_csv(output_path, index=False)

print(f"\n💾 Saved updated dataset to: {output_path}")

# === 5. PREVIEW =============================================================

print("\n📊 Preview of new columns:")
print(points_gdf.head())
