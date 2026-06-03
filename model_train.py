from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ফোল্ডার তৈরি
os.makedirs("models", exist_ok=True)

# একটু বেশি এবং বৈচিত্র্যময় ডামি ডেটা (ধরে নিচ্ছি ইনপুটগুলো হলো: বৃষ্টিপাত, নদীর উচ্চতা, তাপমাত্রা)
# কম ভ্যালু = Low Risk (0), বেশি ভ্যালু = High Risk (1)
X = [
    [10, 1, 25], [15, 2, 24], [20, 1, 26],  # Low Risk ডেটা
    [90, 5, 32], [100, 6, 30], [85, 4, 29], # High Risk ডেটা
    [12, 2, 23], [95, 5, 31], [18, 1, 25]   # মিক্সড ডেটা
]
y = [0, 0, 0, 1, 1, 1, 0, 1, 0] # ০ এবং ১ এর সংখ্যা সামঞ্জস্যপূর্ণ করা হলো

# মডেল ট্রেইনিং
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# মডেল সেভ
joblib.dump(model, "models/flood_model.pkl")

print("Model created successfully with balanced data!")