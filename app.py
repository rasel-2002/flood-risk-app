from flask import Flask, render_template, request
import json
import requests
import os
from dotenv import load_dotenv
import joblib
from sklearn.preprocessing import StandardScaler

load_dotenv()

app = Flask(__name__)
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Load model and scaler
rf_model = joblib.load("models/flood_model.pkl")
scaler = joblib.load("models/scaler.pkl")

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

    # Scale input data
    input_data = [[rainfall, river_level, temperature]]
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0]
    
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    risk_percentage = int(probability[1] * 100)

    result = {
        "risk_level": risk_level,
        "risk_percentage": risk_percentage,
        "area": area_name or "Custom Input",
        "rainfall": rainfall,
        "river_level": river_level,
        "temperature": temperature,
        "weather_desc": weather_desc,
        "source": source,
        "rain_6h": int(min((rainfall / 3) * 0.7, 100)),
        "rain_12h": int(min((rainfall / 3) * 0.9, 100)),
        "rain_24h": int(min(rainfall / 3, 100)),
        "estimated_rain": round(rainfall * 0.3, 1),
        "forecast_date": data.get("forecast_date", ""),
        "forecast_time": data.get("forecast_time", "")
    }

    return render_template("index.html",
                           areas=[a["name"] for a in areas],
                           result=result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
