import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Initializing Rain Classifier training...")

# 1. Load data or generate a synthetic training set if CSV doesn't exist yet
try:
    df = pd.read_csv("weather_logs.csv")
    print("Loaded existing weather_logs.csv dataset.")
except FileNotFoundError:
    print("weather_logs.csv not found. Generating sample training dataset...")
    np.random.seed(42)
    n_samples = 2000
    df = pd.DataFrame({
        'temperature_c': np.random.uniform(15.0, 38.0, n_samples),
        'humidity_pct': np.random.uniform(30.0, 98.0, n_samples),
        'pressure_hpa': np.random.uniform(990.0, 1025.0, n_samples),
        'rain_raw': np.random.randint(0, 4095, n_samples)
    })
    # Realistic heuristic labels for training mock data
    df['pressure_delta'] = df['pressure_hpa'].diff().fillna(0)
    df['is_raining'] = ((df['humidity_pct'] > 76.0) & (df['pressure_delta'] < -0.2)).astype(int)
    df.to_csv("weather_logs.csv", index=False)

# 2. Feature Engineering
df['pressure_delta'] = df['pressure_hpa'].diff().fillna(0)
df.dropna(inplace=True)

X = df[['humidity_pct', 'pressure_delta', 'temperature_c', 'rain_raw']]
y = df['is_raining']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluation
preds = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")

# 6. Save Model
joblib.dump(clf, "rain_classifier.pkl")
print("Success! Model saved to rain_classifier.pkl")