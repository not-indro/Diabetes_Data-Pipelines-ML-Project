from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("model/readmission_model.joblib")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(records: list[dict]):
    df = pd.DataFrame(records)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    return [
        {"prediction": int(p), "risk_score": float(prob)}
        for p, prob in zip(preds, probs)
    ]
