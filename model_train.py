from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

X = [[100, 5, 30], [20, 2, 35], [80, 6, 28]]
y = [1, 0, 1]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/flood_model.pkl")

print("Model created successfully")