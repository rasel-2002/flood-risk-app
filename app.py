from flask import Flask, render_template, request
from models.flood_area import FloodRiskModel
import json

app = Flask(__name__)
model = FloodRiskModel()

def load_areas():
    with open("data/areas.json") as f:
        return json.load(f)["areas"]

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

    if area_data:
        rainfall     = area_data["rainfall"]
        river_level  = area_data["river_level"]
        temperature  = area_data["temperature"]
    else:
        rainfall     = float(data.get("rainfall", 100))
        river_level  = float(data.get("river_level", 5))
        temperature  = float(data.get("temperature", 30))

    result = model.predict(rainfall, river_level, temperature)
    result["area"]        = area_name or "Custom Input"
    result["rainfall"]    = rainfall
    result["river_level"] = river_level
    result["temperature"] = temperature

    return render_template("index.html",
                           areas=[a["name"] for a in areas],
                           result=result)

if __name__ == "__main__":
    app.run(debug=True)