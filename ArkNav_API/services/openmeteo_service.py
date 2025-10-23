import requests

BASE_URL = "https://api.open-meteo.com/v1/forecast"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
TIMEZONE = "Asia%2FSingapore"

# --------------- #
# 🧩 MAIN FUNCTION #
# --------------- #
def get_weather_data(lat, lon):
    """Fetch all weather data (current, hourly, daily) for the given coordinates."""
    return {
        "elevation": get_elevation(lat, lon),
        "current": get_current_weather(lat, lon),
        "daily": get_daily_weather(lat, lon),
        "hourly": get_hourly_weather(lat, lon)       
    }

# -------------------- #
# 🌤 CURRENT CONDITIONS #
# -------------------- #
def get_current_weather(lat, lon):
    """Fetch current weather data such as rain, precipitation, and weather code."""
    url = (
        f"{BASE_URL}?latitude={lat}&longitude={lon}"
        f"&current=precipitation"
        f"&timezone={TIMEZONE}"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("current", {})

# ---------------- #
# ⏰ HOURLY FORECAST #
# ---------------- #
def get_hourly_weather(lat, lon):
    """Fetch hourly forecast data like temperature and rain probability."""
    url = (
        f"{BASE_URL}?latitude={lat}&longitude={lon}"
        f"&hourly=precipitation_probability"
        f"&timezone={TIMEZONE}"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("hourly", {})

# ---------------- #
# 📅 DAILY FORECAST #
# ---------------- #
def get_daily_weather(lat, lon):
    """Fetch daily forecast data such as total precipitation and average probability."""
    url = (
        f"{BASE_URL}?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,precipitation_probability_mean"
        f"&timezone={TIMEZONE}"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("daily", {})

# ---------------- #
# 🏔 ELEVATION DATA #
# ---------------- #
def get_elevation(lat, lon):
    """Fetch elevation data for the coordinates."""
    url = f"{ELEVATION_URL}?latitude={lat}&longitude={lon}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data.get("elevation", [None])[0] if "elevation" in data else None
