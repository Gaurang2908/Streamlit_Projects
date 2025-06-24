import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("dummy_healthcare_data.csv")
df_encoded = pd.get_dummies(df, columns=["gender", "smoker"], drop_first=True)

X = df_encoded.drop(columns=["patient_id", "readmitted_30_days"])
y = df_encoded["readmitted_30_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "readmission_model.pkl")
print("✅ Model saved as readmission_model.pkl")
