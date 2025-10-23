from flask import Flask, request, jsonify
from flask_cors import CORS
from services.openmeteo_service import get_weather_data, get_elevation
from services.datasets_service import check_flood_zone, get_historical_floods, check_nearest_waterway
from services.floodrisk_service import predict_current_flood_risk, predict_sevendays_flood_risk
from collections import OrderedDict

app = Flask(__name__)
CORS(app)  # Allow Flutter to connect

@app.route('/')
def home():
    return jsonify({"message": "ArkNav Flask API is running!"})

@app.route('/predict_flood_risk', methods=['POST'])
def predict_flood_risk_endpoint():
    dataArray = request.json
    result = []
    for data in dataArray:
        locationName = data['locationName']
        lat = data['lat']
        lon = data['lon']

        # Use the trained model for prediction
        current_flood_risk_result = predict_current_flood_risk(lat, lon)
        seven_days_flood_risk_result = predict_sevendays_flood_risk(lat, lon)  # Placeholder for 7-day forecast
        prediction_result = OrderedDict([
            ("locationName", locationName),
            ("current_flood_risk", current_flood_risk_result),
            ("seven_days_flood_risk", seven_days_flood_risk_result)
        ])
        result.append(prediction_result)
    return jsonify(result)


@app.route('/check_flood_zone', methods=['GET'])
def check_flood_zone_endpoint():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    is_in_flood_zone = check_flood_zone(lat, lon)
    return is_in_flood_zone

@app.route('/check_nearest_waterway', methods=['GET'])
def check_nearest_waterway_endpoint():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    waterway_data = check_nearest_waterway(lat, lon)
    return jsonify(waterway_data)

@app.route('/get_historical_floods', methods=['GET'])
def get_historical_floods_endpoint():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    history_data = get_historical_floods(lat, lon)
    return jsonify({"historical_floods": history_data})

@app.route('/get_weather_data', methods=['GET'])
def get_weather_data_endpoint():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    weather_data = get_weather_data(lat, lon)
    return jsonify({"weather_data": weather_data})

@app.route('/get_elevation', methods=['GET'])
def get_elevation_endpoint():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    elevation = get_elevation(lat, lon)
    return jsonify({"elevation": elevation})

@app.route('/get_all_data', methods=['GET'])
def get_all_data_endpoint():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))

    weather_data = get_weather_data(lat, lon)
    flood_zone = check_flood_zone(lat, lon)
    history_data = get_historical_floods(lat, lon)

    return jsonify({
        "weather_data": weather_data,
        "flood_zone": flood_zone,
        "historical_floods": history_data
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


def get_weather_data(lat, lon):
    weather_data = get_weather_data(lat, lon)
    return weather_data

def check_flood_zone(lat, lon, pin_lat, pin_lon):
    flood_zone = check_flood_zone(lat, lon)
    return flood_zone
def get_historical_floods(lat, lon, pin_lat, pin_lon):
    history_data = get_historical_floods(lat, lon)
    return history_data

def predict_flood_risk(weather_data, flood_zone, history_data,):
    flood_risk = predict_flood_risk(weather_data, flood_zone, history_data)
    return flood_risk