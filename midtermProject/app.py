# app.py
# FastAPI service for the diabetes screening model.
#
#   uvicorn app:app --reload      (run train.py first)
#   docs page: http://localhost:8000/docs
#
# All fields are optional. A missing field is ignored and shows up
# in "missing_fields" and "warnings" in the response.

import json
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from features import RAW_FEATURES
from predictor import Predictor

HERE = os.path.dirname(os.path.abspath(__file__))
predictor = Predictor(os.path.join(HERE, "model.joblib"))

app = FastAPI(title="Diabetes screening API", version="1.0")


class Patient(BaseModel):
    # units: height/waist/hip in inches, weight in pounds
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        json_schema_extra={"example": {
            "age": 55, "gender": "male", "height": 68, "weight": 210,
            "waist": 42, "hip": 40, "frame": "large", "bp.1s": 140, "bp.1d": 90,
            "chol": 220, "hdl": 38, "ratio": 5.8, "time.ppn": 300}})

    chol: Optional[float] = None
    hdl: Optional[float] = None
    ratio: Optional[float] = None
    age: Optional[float] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    bp_1s: Optional[float] = Field(None, alias="bp.1s")
    bp_1d: Optional[float] = Field(None, alias="bp.1d")
    waist: Optional[float] = None
    hip: Optional[float] = None
    time_ppn: Optional[float] = Field(None, alias="time.ppn")
    gender: Optional[str] = None
    frame: Optional[str] = None


class Prediction(BaseModel):
    probability_diabetic: float
    threshold: float
    prediction: int
    label: str
    missing_fields: List[str]
    warnings: List[str]


@app.get("/health")
def health():
    return {"status": "ok",
            "threshold": round(predictor.threshold, 4),
            "features": RAW_FEATURES}


@app.get("/samples")
def samples():
    # 2 test + 2 train examples that train.py saved, with their true labels
    with open(os.path.join(HERE, "samples.json")) as f:
        return json.load(f)


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    return predictor.predict(patient.model_dump(by_alias=True))
