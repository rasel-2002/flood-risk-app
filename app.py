from flask import Flask, render_template, request
from models.flood_area import FloodRiskModel
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
model = FloodRiskModel()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

def load_areas():
    with open("data/areas.json") as f:
        return json.load(f)["areas"]

def get_weather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},BD&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") == 200:
            return {
                "rainfall": data.get("rain", {}).get("1h", 0) * 10,
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "weather": data["weather"][0]["description"],
                "success": True
            }
    except:
        pass
    return {"success": False}

@app.route("/")
def home():
    areas = load_areas()
    area_names = [a["name"] for a in areas]
    return render_template("index.html", areas=area_names)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    area_name = data.get("area_name", "")

    areas = load_areas()
    area_data = next((a for a in areas if a["name"] == area_name), None)

    weather = get_weather(area_name)

    if weather["success"]:
        rainfall = weather["rainfall"]
        temperature = weather["temperature"]
        river_level = area_data["river_level"] if area_data else float(data.get("river_level", 5))
        weather_desc = weather["weather"]
        source = "🌐 Real-time weather"
    elif area_data:
        rainfall = area_data["rainfall"]
        temperature = area_data["temperature"]
        river_level = area_data["river_level"]
        weather_desc = "Historical data"
        source = "📊 Historical data"
    else:
        rainfall = float(data.get("rainfall", 100))
        temperature = float(data.get("temperature", 30))
        river_level = float(data.get("river_level", 5))
        weather_desc = "Manual input"
        source = "✏️ Manual input"

    result = model.predict(rainfall, river_level, temperature)
    result["area"] = area_name or "Custom Input"
    result["rainfall"] = rainfall
    result["river_level"] = river_level
    result["temperature"] = temperature
    result["weather_desc"] = weather_desc
    result["source"] = source
    # Rain possibility calculation
    base = min(rainfall / 3, 100)
    result["rain_6h"] = int(min(base * 0.7, 100))
    result["rain_12h"] = int(min(base * 0.9, 100))
    result["rain_24h"] = int(min(base, 100))
    result["estimated_rain"] = round(rainfall * 0.3, 1)
    result["forecast_date"] = data.get("forecast_date", "")
    result["forecast_time"] = data.get("forecast_time", "")

    return render_template("index.html",
                           areas=[a["name"] for a in areas],
                           result=result)

if __name__ == "__main__":
    app.run(debug=True)