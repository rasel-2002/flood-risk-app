from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("models", exist_ok=True)

# বেশি এবং realistic training data
X = [
    # Low Risk (0)
    [50, 2, 25], [60, 3, 26], [70, 3.5, 27], [55, 2.5, 24],
    [65, 3, 28], [45, 2, 23], [75, 3.8, 29], [40, 1.5, 22],
    
    # High Risk (1)
    [250, 7, 32], [280, 8, 33], [300, 9, 34], [320, 9.8, 36],
    [290, 8.5, 35], [270, 7.8, 33], [310, 9.2, 35], [260, 7.5, 32],
    
    # Medium Risk (0-1 boundary)
    [150, 5, 30], [180, 6, 31], [120, 4.5, 29], [200, 6.5, 32]
]

y = [0, 0, 0, 0, 0, 0, 0, 0,  # Low Risk
     1, 1, 1, 1, 1, 1, 1, 1,  # High Risk
     0, 0, 0, 0]               # Medium (treated as Low)

# Model training
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "models/flood_model.pkl")

print("✅ Model trained successfully with realistic data!")
