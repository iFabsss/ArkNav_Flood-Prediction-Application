import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points
import math

# --- Load once at startup ---
flood_zones = gpd.read_file('data/MetroManila 5 Years Project NOAH/MetroManila_Flood_5year.shp')
flood_history = pd.read_csv('data/AEGISDataset_NCR.csv')
ph_waterways = gpd.read_file('data/phl_rivl_250k_namria/phl_rivl_250k_NAMRIA.shp')

# Ensure flood zones are in EPSG:3857 (meters)
if flood_zones.crs.to_epsg() != 3857:
    flood_zones = flood_zones.to_crs(epsg=3857)

# --- 1️⃣ Check flood hazard zone ---
def check_flood_zone(lat, lon, radius_m=300):
    # Convert user point to EPSG:3857
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)

    # 300 m radius buffer (no degree conversion needed)
    buffer = point.buffer(radius_m)

    # Find all flood zones intersecting with buffer
    nearby_zones = flood_zones[flood_zones.intersects(buffer.iloc[0])]

    if nearby_zones.empty:
        return {
            "lat": lat,
            "lon": lon,
            "in_zone": False,
            "hazard_code": None,
            "hazard_level": None,
            "nearby_zones": []
        }

    hazard_map = {
        1: "Low (0–0.5m)",
        2: "Medium (>0.5–1.5m)",
        3: "High (>1.5m)"
    }

    # Convert back to degrees to check exact containment
    point_deg = point.to_crs(epsg=4326)
    matching_zones = nearby_zones.to_crs(epsg=4326)
    matching_zones = matching_zones[matching_zones.contains(point_deg.iloc[0])]

    if not matching_zones.empty:
        hazard_code = int(matching_zones.iloc[0]['Var'])
        hazard_description = hazard_map.get(hazard_code, "Unknown")
        in_zone = True
    else:
        hazard_code = None
        hazard_description = None
        in_zone = False

    nearby_zones_list = [
        {
            "hazard_code": int(z['Var']),
            "hazard_level": hazard_map.get(int(z['Var']), "Unknown")
        }
        for _, z in nearby_zones.iterrows()
    ]

    return {
        "lat": lat,
        "lon": lon,
        "in_zone": in_zone,
        "hazard_code": hazard_code,
        "hazard_level": hazard_description,
        "nearby_zones": nearby_zones_list
    }

# Ensure it uses EPSG:32651 (NAMRIA standard)
if ph_waterways.crs.to_epsg() != 32651:
    ph_waterways = ph_waterways.to_crs(epsg=32651)

# --- Helper: Convert meters between CRS ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (m) between two lat/lon points."""
    R = 6371000  # radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- 🌊 3️⃣ Get nearest waterway ---
def check_nearest_waterway(lat, lon, search_radius_m=2000):
    """
    Finds the nearest river/waterway to given coordinates.
    Returns river name, coordinates, and distance (m).
    """

    # Convert the point to GeoSeries with EPSG:4326
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")

    # Reproject to same CRS as rivers (EPSG:32651)
    point_proj = point.to_crs(epsg=32651)

    # Create a spatial buffer for efficient querying
    buffer = point_proj.buffer(search_radius_m).iloc[0]
    nearby_rivers = ph_waterways[ph_waterways.intersects(buffer)]

    # If no nearby rivers within radius, expand the search
    if nearby_rivers.empty:
        nearby_rivers = ph_waterways.copy()

    # Compute nearest river using geometric distance in projected CRS
    distances = nearby_rivers.distance(point_proj.iloc[0])
    nearest_idx = distances.idxmin()
    nearest_geom = nearby_rivers.loc[nearest_idx, 'geometry']

    # Get the nearest point along the river line
    nearest_point_geom = nearest_points(point_proj.iloc[0], nearest_geom)[1]

    # Convert nearest point back to lat/lon
    nearest_point_wgs84 = gpd.GeoSeries([nearest_point_geom], crs=ph_waterways.crs).to_crs(epsg=4326).iloc[0]

    # Compute haversine distance for accurate metric value
    nearest_lat, nearest_lon = nearest_point_wgs84.y, nearest_point_wgs84.x
    dist_m = haversine_distance(lat, lon, nearest_lat, nearest_lon)

    # River name handling
    river_name = nearby_rivers.loc[nearest_idx, 'RIVER_NAME']
    if not isinstance(river_name, str) or river_name.strip() == "":
        river_name = "Unnamed Waterway"

    return {
        "lat": lat,
        "lon": lon,
        "nearest_river": river_name,
        "nearest_river_lat": nearest_lat,
        "nearest_river_lon": nearest_lon,
        "dist_to_nearest_water_m": dist_m
    }

# --- 2️⃣ Get historical flood data ---
def get_historical_floods(lat, lon, radius_m=300):
    """
    Gets flood history records within 300 m using simple bounding box.
    Converts meter radius → degrees for filtering.
    """
    # Convert 300 m to degrees (~111 km per degree)
    radius_deg = radius_m / 111_000

    subset = flood_history[
        (flood_history['lat'].between(lat - radius_deg, lat + radius_deg)) &
        (flood_history['lon'].between(lon - radius_deg, lon + radius_deg))
    ].copy()

    flood_descriptions = {
        0: "No flood",
        1: "Ankle High",
        2: "Knee High",
        3: "Waist High",
        4: "Neck High",
        5: "Top of Head High",
        6: "1-storey High",
        7: "1.5-storey High",
        8: "2-storeys or Higher"
    }

    subset['flood_description'] = subset['flood_height'].map(flood_descriptions)

    return subset.to_dict(orient='records')
