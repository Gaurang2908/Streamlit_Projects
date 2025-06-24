import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# === Load Data ===
df = pd.read_csv(r"dummy_healthcare_data.csv")  

# === Preprocessing ===
possible_targets = [col for col in df.columns if "readmit" in col.lower()]
if not possible_targets:
    raise ValueError("Could not find a target column related to 'readmit'")
target_col = possible_targets[0]
print(f"Target column detected: '{target_col}'")


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
# Resample with SMOTE to fix class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model on resampled data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train with XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    scale_pos_weight=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_resampled, y_resampled)


# === Evaluate ===
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# === Save Trained Model ===
joblib.dump(model, "readmission_model.pkl")
print("✅ Model saved as 'readmission_model.pkl'")
