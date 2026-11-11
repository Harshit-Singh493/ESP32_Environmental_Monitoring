import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Initializing Rain Classifier training...")

try:
    df = pd.read_csv("weather_logs.csv")
    print(f"Loaded existing weather_logs.csv dataset with {len(df)} rows.")
except FileNotFoundError:
    print("weather_logs.csv not found. Please run export_data.py first.")
    exit()

# 1. Feature Engineering: Calculate pressure delta
df['pressure_delta'] = df['pressure_hpa'].diff().fillna(0)

# 2. Create the missing target label
# Since the DB only stores raw analog values, we generate the binary label here:
df['is_raining'] = ((df['humidity_pct'] > 75.0) & (df['pressure_delta'] < -0.2)).astype(int)

# 3. Drop NaNs (using the modern Pandas 3.x syntax to fix your warning)
df = df.dropna()

X = df[['humidity_pct', 'pressure_delta', 'temperature_c', 'rain_raw']]
y = df['is_raining']

# 4. Handle the "No Rain Yet" edge case
if len(y.unique()) < 2:
    print("Notice: Your real data only contains dry weather so far.")
    print("Injecting a few synthetic rainy rows so the classifier can compile...")
    
    synthetic_rain = pd.DataFrame({
        'humidity_pct': [90.0, 95.0, 88.0, 92.0],
        'pressure_delta': [-1.5, -2.0, -0.8, -1.2],
        'temperature_c': [22.0, 21.5, 23.0, 22.8],
        'rain_raw': [1500, 1200, 1800, 1400],
        'is_raining': [1, 1, 1, 1]
    })
    
    df = pd.concat([df, synthetic_rain], ignore_index=True)
    X = df[['humidity_pct', 'pressure_delta', 'temperature_c', 'rain_raw']]
    y = df['is_raining']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluation
preds = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")

# 8. Save Model
joblib.dump(clf, "rain_classifier.pkl")
print("Success! Model saved to rain_classifier.pkl")