# retrain_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === Load Data ===
df = pd.read_csv("your_updated_health_data.csv")  # ← Replace with your actual CSV

# === Preprocessing ===
target_col = "Readmissions within 30 days"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in data.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target if needed
if y.dtype == "object":
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)
    joblib.dump(y_le, "target_encoder.pkl")  # Optional: Save target encoder

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# === Save Trained Model ===
joblib.dump(model, "readmission_model.pkl")
print("✅ Model saved as 'readmission_model.pkl'")
