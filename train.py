# File: train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # for saving the model
import matplotlib.pyplot as plt

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

plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.ylim(0, 1) # Set Y-axis limit from 0 to 1 for accuracy
plt.title('Model Accuracy')
plt.ylabel('Accuracy Score')
# Add the score text on the bar
plt.text(0, accuracy + 0.02, f'{accuracy:.4f}', ha='center') 
plt.savefig('accuracy_chart.png') # 💡 Saves the visualization as a PNG file
plt.close() # Close the figure to free up memory

print(f"Model trained and saved. Accuracy: {accuracy:.4f}")
