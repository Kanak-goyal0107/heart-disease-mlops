
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps project using a trained Random Forest model",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

class HeartDiseaseInput(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: str
    ChestPainType: str
    RestingBP: int = Field(ge=0)
    Cholesterol: int = Field(ge=0)
    FastingBS: int = Field(ge=0, le=1)
    RestingECG: str
    MaxHR: int = Field(ge=0, le=250)
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        return {
            "prediction": prediction,
            "label": (
                "Heart disease detected by model"
                if prediction == 1
                else "No heart disease predicted by model"
            ),
            "probability_class_0": round(float(probabilities[0]), 4),
            "probability_class_1": round(float(probabilities[1]), 4),
            "note": "Educational prediction only; not medical advice.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc
