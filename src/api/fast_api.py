# fast_api.py

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.predict_pipeline import (PredictPipeline, PatientData)
from src.logger import logger

# --- App Setup ------------------------------------------
app = FastAPI(
    title       = "Health Risk Predictor API",
    description = """
    Predicts diabetes risk from patient health indicators.
    Trained on CDC BRFSS 2015 dataset (253,680 patients).
    Model: XGBoost | ROC-AUC: 0.8242 | Recall: 0.79
    """,
    version     = "1.0.0"
)

# ── Environment-based CORS origins ────────────────────────
# Development: allow localhost ports
# Production : only allow your actual deployed domain

ENV = os.getenv('APP_ENV', 'development')

if ENV == 'production':
    allowed_origins = [
        "https://healthrisk.streamlit.app",  # your Streamlit app
        "https://yourdomain.com",             # your custom domain
    ]
else:
    # Development only
    allowed_origins = [
        "http://localhost:8501",   # Streamlit default port
        "http://localhost:3000",   # React dev server
        "http://localhost:5000",   # Flask dev server
        "http://127.0.0.1:8501",
        "http://127.0.0.1:5000",
    ]

# --- Cross-Origin Resource Sharing (CORS) - allows Streamlit to call this API ---------------
app.add_middleware(
    CORSMiddleware,
    allow_origins     = allowed_origins,  # ← specific list
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],  # ← only what we need
    allow_headers     = ["Content-Type", "Authorization"],
)

# --- Load pipeline once at startup --------------------------------------------------------
pipeline = PredictPipeline()

# --- Request schema (Pydantic validates input)-----------------------------------
class PatientInput(BaseModel):
    """
    Patient health indicators.
    FastAPI auto-validates types and ranges.
    
    Args:
        BaseModel (_type_): _description_
    """
    HighBP              : int   = Field(..., ge=0, le=1, description="High blood pressure (0=No, 1=Yes)")
    HighChol            : int   = Field(..., ge=0, le=1, description="High cholesterol (0=No, 1=Yes)")
    BMI                 : float = Field(..., ge=10, le=100, description="Body Mass Index ")
    Smoker              : int   = Field(..., ge=0, le=1, description="Smoked 100+ cigarettes (0=No, 1=Yes)")
    Stroke              : int   = Field(..., ge=0, le=1, description="Ever had stroke (0=No, 1=Yes)")
    HeartDiseaseorAttack: int   = Field(..., ge=0, le=1, description="Heart disease history (0=No, 1=Yes)")
    PhysActivity        : int   = Field(..., ge=0, le=1, description="Physical activity in past 30 days (0=No, 1=Yes)")
    Fruits              : int   = Field(..., ge=0, le=1, description="Eat fruit daily (0=No, 1=Yes)")
    Veggies             : int   = Field(..., ge=0, le=1, description="Eat vegetables daily (0=No, 1=Yes)")
    HvyAlcoholConsump   : int   = Field(..., ge=0, le=1, description="Heavy drinker (0=No, 1=Yes)")
    GenHlth             : int   = Field(..., ge=1, le=5, description="General health (1=Excellent, 5 = Poor)")
    MentHlth            : int   = Field(..., ge=0, le=30, description="Bad mental health (0-30)")
    PhysHlth            : int   = Field(..., ge=0, le=30, description="Bad physical health (0-30)")
    DiffWalk            : int   = Field(..., ge=0, le=1, description="difficulty walking (0=No, 1=Yes)")
    Age                 : int   = Field(..., ge=1, le=13, description="Age group 1 = 18-24 to 13 = 80+")
    Education           : int   = Field(..., ge=1, le=6, description="Education level 1 = None to 6 = College")
    Income              : int   = Field(..., ge=1, le=8, description="Income level 1=<$10K to 8=$75K+")
    
    class Config:
        jason_schema_extra = {
            "example":{
                "HighBP": 1, "HighChol": 1, "BMI": 35.0,
                "Smoker": 1, "Stroke": 0,
                "HeartDiseaseorAttack": 1,
                "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
                "HvyAlcoholConsump": 0, "GenHlth": 4,
                "MentHlth": 10, "PhysHlth": 15,
                "DiffWalk": 1, "Age": 10,
                "Education": 3, "Income": 2
            }
        }

# --- Response Schema --------------------------------
class PredictionResponce(BaseModel):
    prediction   : int
    probability  : float
    risk_level   : str
    label        : str
    risk_color   : str
    
    
# --- Routes ------------------------------------------
@app.get("/")
def root():
    """
    Root endpoint - API info.
    """
    return {
        "message" : "Health Risk Predictor API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "predict" : "/predict"
    }
    
@app.get("/health")
def health_check():
    """
    Health check for deployment monitoring.
    """
    return {
        "Status"    : "healthy",
        "model"     : "XGBoost Diabetes Predictor",
        "roc_auc"   : 0.8242,
        "version"   : "1.0.0"
    }
    
@app.post("/predict", response_model=PredictionResponce)
def predict(patient_input: PatientInput):
    """
    Predicts diabetes risk for a patient.
    
    Args:
        patient_input (PatientInput): _description_
        
    Returns:
        prediction  : 0 = No Diabetes, 1 = Diabetes
        probability : Risk probability (0-100%)
        risk_level  : Low / Moderate / High
        label       : Human readable result
        risk_color  : Green / Orange / Red
    """
    try:
        logger.info(" API /predict call")
        
        #Convert Pydantic model to PatientData
        patient = PatientData(**patient_input.model_dump())
        
        # Run prediction
        result = pipeline.predict(patient)
        
        logger.info(f"API response: {result['label']}" 
                    f"({result['probability']}%)"
            )
        return result
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
