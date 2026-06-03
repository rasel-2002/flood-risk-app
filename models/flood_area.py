import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class FloodRiskModel:
    def __init__(self):
        self.model_path = "models/flood_model.pkl"
        self.model = self._load_or_train()

    def _train_model(self):
        X = np.array([
            [200, 8.5, 35], [150, 6.0, 30], [300, 9.5, 38],
            [50,  2.0, 25], [80,  3.5, 27], [250, 8.0, 33],
            [400, 10.0, 40],[30,  1.5, 22], [180, 7.0, 31],
            [100, 4.5, 28],
        ])
        y = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.model_path)
        return model

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return self._train_model()

    def predict(self, rainfall, river_level, temperature):
        features = np.array([[rainfall, river_level, temperature]])
        result = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 1)

        if result == 1:
            return {"level": "HIGH", "confidence": confidence,
                    "color": "danger", "icon": "⚠️"}
        else:
            return {"level": "LOW", "confidence": confidence,
                    "color": "success", "icon": "✅"}