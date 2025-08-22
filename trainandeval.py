# trainandeval.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import argparse

# ===============================
# Command line argument
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help="Use full dataset (slow)")
args = parser.parse_args()

# ===============================
# Load Dataset
# ===============================
data_file = "covtype.data.gz"

columns = [
    'Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3',
    'Wilderness_Area4'
] + [f'Soil_Type{i}' for i in range(1, 41)] + ['Cover_Type']

if not os.path.exists(data_file):
    raise FileNotFoundError(f"{data_file} not found.")

# Read as plain text (not actually gzipped)
df = pd.read_csv(data_file, header=None, names=columns)

# ===============================
# Optional: sample for testing
# ===============================
if not args.full:
    df = df.sample(20000, random_state=42)
    print("Using 20k rows for fast testing.")

X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Train Random Forest
# ===============================
print("Training Random Forest...")
best_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train)

# ===============================
# Train XGBoost (fix labels 1-7 -> 0-6)
# ===============================
print("Training XGBoost...")
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

best_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
best_xgb.fit(X_train, y_train_xgb)

# ===============================
# Evaluate Models
# ===============================
models = {
    "RandomForest": (best_rf, y_test, None),
    "XGBoost": (best_xgb, y_test, y_pred_offset := 1)  # +1 to revert 0-6 -> 1-7
}

results = {}

for name, (model, y_true, offset) in models.items():
    if name == "XGBoost":
        y_pred = model.predict(X_test) + offset
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    results[name] = acc

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

# Accuracy comparison
plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.close()

# Save models
joblib.dump(best_rf, "random_forest_model.joblib")
joblib.dump(best_xgb, "xgb_model.joblib")
print("\n✅ Models trained and saved successfully!")
