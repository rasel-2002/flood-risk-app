from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

os.makedirs("models", exist_ok=True)

# বেশি realistic এবং balanced training data
X = [
    # Low Risk (0) - কম বৃষ্টি, কম নদীর উচ্চতা
    [30, 1.5, 22], [40, 2, 23], [50, 2.5, 24], [45, 2, 25],
    [35, 1.8, 23], [55, 2.8, 26], [60, 3, 27], [48, 2.3, 24],
    [42, 2.1, 25], [52, 2.6, 26],
    
    # Medium Risk (0) - মাঝারি বৃষ্টি, মাঝারি নদীর উচ্চতা
    [100, 4, 28], [120, 4.5, 29], [110, 4.2, 28], [130, 4.8, 30],
    [105, 4.1, 29], [125, 4.6, 29], [115, 4.3, 28], [135, 4.9, 30],
    
    # High Risk (1) - বেশি বৃষ্টি, বেশি নদীর উচ্চতা
    [250, 7, 32], [280, 8, 33], [300, 9, 34], [320, 9.8, 36],
    [270, 7.5, 32], [290, 8.5, 35], [310, 9.2, 35], [260, 7.2, 33],
    [275, 7.8, 33], [305, 9.1, 35]
]

y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Low Risk (10)
     0, 0, 0, 0, 0, 0, 0, 0,        # Medium Risk (8) - treated as Low
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # High Risk (10)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training with better parameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_scaled, y)

# Save both model and scaler
joblib.dump(model, "models/flood_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model trained successfully with balanced and scaled data!")
