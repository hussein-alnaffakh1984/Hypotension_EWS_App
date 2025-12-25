import joblib

# افترض أن لديك:
# model = trained LogisticRegression
# scaler = fitted StandardScaler

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler exported successfully.")
