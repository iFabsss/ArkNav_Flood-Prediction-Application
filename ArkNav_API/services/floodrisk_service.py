import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

from services.openmeteo_service import get_weather_data
from services.datasets_service import check_flood_zone, check_nearest_waterway, flood_zones

# --- Load model and scaler once ---
MODEL_PATH = "models/Arknav_Floodrisk_Modelv2.pkl"
model_pkg = joblib.load(MODEL_PATH)
flood_model = model_pkg["model"]
scaler = model_pkg["scaler"]

# Manila center for distance reference
MANILA_CENTER_LAT, MANILA_CENTER_LON = 14.5906, 120.9735


# --- 📊 FEATURE EXTRACTION ---
def get_flood_features(lat, lon, forecast_type):
    """Collects all features needed for the model, matching training pipeline."""

    # 1️⃣ Get elevation & weather
    weather_data = get_weather_data(lat, lon)
    elevation = weather_data.get("elevation", 0)
    precipitation = 0
    if "current" in weather_data and "precipitation" in weather_data["current"]:
        precipitation = weather_data["current"]["precipitation"]
    elif "daily" in weather_data and "precipitation_sum" in weather_data["daily"]:
        precipitation = weather_data["daily"]["precipitation_sum"][0]

    # 2️⃣ Flood zone info
    flood_info = check_flood_zone(lat, lon)
    in_zone_int = int(flood_info.get("in_zone", False))

    # Ensure hazard_code is numeric
    hazard_code = flood_info.get("hazard_code")
    if hazard_code is None or not isinstance(hazard_code, (int, float)):
        hazard_code = 1

    # 3️⃣ Nearest river info
    water_info = check_nearest_waterway(lat, lon)
    dist_to_nearest_water_m = water_info.get("dist_to_nearest_water_m", 0)
    nearest_river_lat = water_info.get("nearest_water_lat", 0)
    nearest_river_lon = water_info.get("nearest_water_lon", 0)

    # 4️⃣ Proximity to flood zone
    if in_zone_int:
        proximity_m = 0
    else:
        point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        distances = flood_zones.distance(point.iloc[0])
        proximity_m = distances.min() if not distances.empty else 0

    # --- Derived / engineered features ---
    elevation_risk = 1 / (elevation + 1)
    proximity_risk = 1 / (proximity_m + 1)
    river_distance_risk = 1 / (dist_to_nearest_water_m + 1)
    dist_from_center = np.sqrt((lat - MANILA_CENTER_LAT)**2 + (lon - MANILA_CENTER_LON)**2) * 111
    precip_x_elevation = precipitation * elevation_risk
    precip_squared = precipitation ** 2
    zone_x_hazard = in_zone_int * hazard_code
    zone_x_precip = in_zone_int * precipitation
    risk_score = (in_zone_int * 3) + ((precipitation / 10) * 2) + (elevation_risk * 10) + (proximity_risk * 5)
    precip_x_elev_x_prox = precipitation * elevation_risk * proximity_risk
    precipRisk_x_riverdistance = precipitation * dist_to_nearest_water_m

    # Combine everything into a single consistent dictionary
    return {
        "lat": lat,
        "lon": lon,
        "elevation": elevation,
        "precipitation": precipitation,
        "hazard_code": hazard_code,
        "proximity_m": proximity_m,
        "in_zone_int": in_zone_int,
        "elevation_risk": elevation_risk,
        "proximity_risk": proximity_risk,
        "precip_x_elevation": precip_x_elevation,
        "zone_x_precip": zone_x_precip,
        "zone_x_hazard": zone_x_hazard,
        "risk_score": risk_score,
        "precip_squared": precip_squared,
        "dist_from_center": dist_from_center,
        "nearest_river_lon": nearest_river_lon,
        "nearest_river_lat": nearest_river_lat,
        "precipRisk_x_riverdistance": precipRisk_x_riverdistance,
        "precip_x_elev_x_prox": precip_x_elev_x_prox,
        "dist_to_nearest_water_m": dist_to_nearest_water_m,
    }


# --- 🤖 CURRENT DAY FLOOD PREDICTION ---
def predict_current_flood_risk(lat, lon):
    """Predicts flood height and risk category based on engineered features."""

    forecast_type = "current"
    features = get_flood_features(lat, lon, forecast_type)

    # Ensure consistent feature order
    feature_cols = [
        'lat', 
        'lon', 
        'elevation', 
        'precipitation', 
        'hazard_code',
        'proximity_m', 
        'in_zone_int', 
        'elevation_risk', 
        'proximity_risk',
        'precip_x_elevation', 
        'zone_x_precip', 
        'zone_x_hazard', 
        'risk_score',
        'precip_squared', 
        'dist_from_center', 
        'nearest_river_lon',
        'nearest_river_lat', 
        'precipRisk_x_riverdistance',
        'precip_x_elev_x_prox'
    ]

    feature_vector = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], columns=feature_cols)
    feature_vector = feature_vector.fillna(0)

    # Scale features using training scaler
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict flood height
    flood_height = max(0, float(flood_model.predict(feature_vector_scaled)[0]))

    # --- Categorize predicted height ---
    if flood_height < 0:
        category = "No Flood"
    elif flood_height >0 and flood_height < 1.5:
        category = "Ankle High"
    elif flood_height >= 1.5 and flood_height < 2.5:
        category = "Knee High"
    elif flood_height >= 2.5 and flood_height < 3.5:
        category = "Waist High"
    elif flood_height >= 3.5 and flood_height < 4.5:
        category = "Neck High"
    elif flood_height >= 4.5 and flood_height < 5.5:
        category = "Top of Head High"
    elif flood_height >= 5.5 and flood_height < 6.5:
        category = "1-storey High"
    elif flood_height >= 6.5 and flood_height < 8.0:
        category = "1.5-storey High"
    elif flood_height >= 8.0 and flood_height < 10.0:
        category = "2-storeys or Higher"
    else:
        category = "Severe Flooding"

    # --- Get confidence if available ---
    probability = predict_flood_probability(flood_model, scaler, feature_vector)
        

    # --- Return structured result ---
    return {
        "input_coordinates": {"lat": lat, "lon": lon},
        "features": features,
        "predicted_flood_height_m": flood_height,
        "category": category,
        "confidence": probability
    }

# --- 🤖 SEVEN DAYS FLOOD PREDICTION ---
def predict_sevendays_flood_risk(lat, lon):
    """
    Predicts 7-day flood height and risk category for each forecasted day
    using engineered features and the trained model.
    """

    # 1️⃣ Get weather forecast data (7 days)
    weather_data = get_weather_data(lat, lon)
    daily_precip = weather_data.get("daily", {}).get("precipitation_sum", [])
    dates = weather_data.get("daily", {}).get("time", [])
    elevation = weather_data.get("elevation", 0)

    # 2️⃣ Pre-compute static spatial features
    flood_info = check_flood_zone(lat, lon)
    in_zone_int = int(flood_info.get("in_zone", False))

    # Ensure hazard_code is numeric and never None
    hazard_code = flood_info.get("hazard_code")
    if hazard_code is None or not isinstance(hazard_code, (int, float)):
        hazard_code = 1


    water_info = check_nearest_waterway(lat, lon)
    dist_to_nearest_water_m = water_info.get("dist_to_nearest_water_m", 0)
    nearest_river_lat = water_info.get("nearest_water_lat", 0)
    nearest_river_lon = water_info.get("nearest_water_lon", 0)

    # 3️⃣ Compute proximity to flood zone
    if in_zone_int:
        proximity_m = 0
    else:
        point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        distances = flood_zones.distance(point.iloc[0])
        proximity_m = distances.min() if not distances.empty else 0

    # 4️⃣ Prepare static derived features
    elevation_risk = 1 / (elevation + 1)
    proximity_risk = 1 / (proximity_m + 1)
    river_distance_risk = 1 / (dist_to_nearest_water_m + 1)
    dist_from_center = np.sqrt((lat - MANILA_CENTER_LAT)**2 + (lon - MANILA_CENTER_LON)**2) * 111

    # 5️⃣ Prepare full forecasts list
    predictions = []

    for i, precipitation in enumerate(daily_precip):
        date = dates[i] if i < len(dates) else f"Day {i+1}"

        # Derived dynamic features (per day)
        precip_x_elevation = precipitation * elevation_risk
        precip_squared = precipitation ** 2
        zone_x_hazard = in_zone_int * hazard_code
        zone_x_precip = in_zone_int * precipitation
        risk_score = (in_zone_int * 3) + ((precipitation / 10) * 2) + (elevation_risk * 10) + (proximity_risk * 5)
        precip_x_elev_x_prox = precipitation * elevation_risk * proximity_risk
        precipRisk_x_riverdistance = precipitation * dist_to_nearest_water_m

        # Combine all features for that day
        features = {
            "lat": lat,
            "lon": lon,
            "elevation": elevation,
            "precipitation": precipitation,
            "hazard_code": hazard_code,
            "proximity_m": proximity_m,
            "in_zone_int": in_zone_int,
            "elevation_risk": elevation_risk,
            "proximity_risk": proximity_risk,
            "precip_x_elevation": precip_x_elevation,
            "zone_x_precip": zone_x_precip,
            "zone_x_hazard": zone_x_hazard,
            "risk_score": risk_score,
            "precip_squared": precip_squared,
            "dist_from_center": dist_from_center,
            "nearest_river_lon": nearest_river_lon,
            "nearest_river_lat": nearest_river_lat,
            "precipRisk_x_riverdistance": precipRisk_x_riverdistance,
            "precip_x_elev_x_prox": precip_x_elev_x_prox,
            "dist_to_nearest_water_m": dist_to_nearest_water_m
        }

        # Ensure consistent column order
        feature_cols = [
            'lat', 'lon', 'elevation', 'precipitation', 'hazard_code',
            'proximity_m', 'in_zone_int', 'elevation_risk', 'proximity_risk',
            'precip_x_elevation', 'zone_x_precip', 'zone_x_hazard', 'risk_score',
            'precip_squared', 'dist_from_center', 'nearest_river_lon',
            'nearest_river_lat', 'precipRisk_x_riverdistance', 'precip_x_elev_x_prox'
        ]

        feature_vector = pd.DataFrame([[features.get(col, 0) for col in feature_cols]], columns=feature_cols)
        feature_vector = feature_vector.fillna(0)

        # Scale & predict
        feature_vector_scaled = scaler.transform(feature_vector)
        flood_height = max(0, float(flood_model.predict(feature_vector_scaled)[0]))

        # Confidence & category
        probability = predict_flood_probability(flood_model, scaler, feature_vector)
        height = flood_height
        if height <= 0:
            category = "No Flood"
        elif height < 1.5:
            category = "Ankle High"
        elif height < 2.5:
            category = "Knee High"
        elif height < 3.5:
            category = "Waist High"
        elif height < 4.5:
            category = "Neck High"
        elif height < 5.5:
            category = "Top of Head High"
        elif height < 6.5:
            category = "1-storey High"
        elif height < 8.0:
            category = "1.5-storey High"
        elif height < 10.0:
            category = "2-storeys or Higher"
        else:
            category = "Severe Flooding"

        predictions.append({
            "date": date,
            "precipitation_mm": precipitation,
            "predicted_flood_height_m": height,
            "category": category,
            "confidence": probability,
            "features": features
        })

    return {
        "input_coordinates": {"lat": lat, "lon": lon},
        "seven_day_forecast": predictions
    }



def predict_flood_probability(model, scaler, feature_vector, k=3.0, max_height=2.0):
    """
    Computes the probability of flooding given input features using a regression model.
    - model: trained XGBoost regressor
    - scaler: fitted RobustScaler
    - feature_vector: pd.DataFrame with same feature columns
    - k: steepness factor for probability curve
    - max_height: normalization reference (e.g., 2m flood = ~100% probability)
    """

    # Scale features
    scaled = scaler.transform(feature_vector)

    # Predict flood height
    predicted_height = np.maximum(0, model.predict(scaled)[0])

    # Option 1: Logistic scaling
    prob = 1 / (1 + np.exp(-k * (predicted_height - 0.1)))

    # Option 2 (alternative): linear normalization
    prob_linear = min(predicted_height / max_height, 1.0)

    return {
        "predicted_height": float(predicted_height),
        "probability_logistic": float(prob),
        "probability_linear": float(prob_linear)
    }

