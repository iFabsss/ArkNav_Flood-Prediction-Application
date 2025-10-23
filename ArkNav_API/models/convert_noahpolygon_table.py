import geopandas as gpd
import pandas as pd

# 1️⃣ Load the shapefile
gdf = gpd.read_file(r'data\MetroManila 5 Years Project NOAH\MetroManila_Flood_5year.shp')

# 2️⃣ Explode multipolygons to individual polygons
gdf_exploded = gdf.explode(index_parts=False)

# 3️⃣ Store original centroids (in degrees)
gdf_exploded['centroid_lat_deg'] = gdf_exploded.geometry.centroid.y
gdf_exploded['centroid_lon_deg'] = gdf_exploded.geometry.centroid.x

# 4️⃣ Reproject to a meter-based CRS for correct area and length (EPSG:3857)
gdf_meters = gdf_exploded.to_crs(epsg=3857)

# 5️⃣ Compute geometric features in meters
gdf_meters['area_m2'] = gdf_meters.geometry.area
gdf_meters['perimeter_m'] = gdf_meters.geometry.length

# 6️⃣ Combine key attributes into a DataFrame
df = pd.DataFrame({
    'Var': gdf_meters['Var'],
    'centroid_lat': gdf_exploded['centroid_lat_deg'],  # keep geographic for display
    'centroid_lon': gdf_exploded['centroid_lon_deg'],
    'area_m2': gdf_meters['area_m2'],
    'perimeter_m': gdf_meters['perimeter_m']
})

# 7️⃣ Export to CSV
df.to_csv(r'data\MetroManila_5Years_ProjectNOAH_Tabular.csv', index=False)

print("✅ CSV successfully exported with accurate areas and perimeters!")
