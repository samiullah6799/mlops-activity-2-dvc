# File: train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # for saving the model

# --- 1. Load Data ---
try:
    df = pd.read_csv('data/dataset.csv')
    X = df[['feature']]
    y = df['target']
except Exception as e:
    # Handle the case where the data file is empty or missing headers (e.g., in a clean state)
    print("Could not load data. Ensure data/dataset.csv has content.")
    exit(1)

# --- 2. Train Model ---
model = LogisticRegression(random_state=42)
model.fit(X, y)

# --- 3. Evaluate Model ---
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# --- 4. Save Artifacts for DVC/CML ---

# Save the metric
with open('metrics.txt', 'w') as f:
    f.write(f"accuracy: {accuracy:.4f}\n")

# Save the model
joblib.dump(model, 'model.pkl')

print(f"Model trained and saved. Accuracy: {accuracy:.4f}")
