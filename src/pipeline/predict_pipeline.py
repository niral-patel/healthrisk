#predict_pipeline.py

import os
import sys

import numpy as np
import pandas as pd
import joblib

from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.utils import engineer_features


# --- Patient input schema --------------------------------
@dataclass
class PatientData:
    """
    Holds raw input from the web form.
    One field per feature the user fills in.
    All values come in as Python native types.
    """
    HighBP              : int
    HighChol            : int
    BMI                 : float
    Smoker              : int
    Stroke              : int
    HeartDiseaseorAttack: int
    PhysActivity        : int
    Fruits              : int
    Veggies             : int
    HvyAlcoholConsump   : int
    GenHlth             : int
    MentHlth            : int
    PhysHlth            : int
    DiffWalk            : int
    Age                 : int
    Education           : int
    Income              : int 
    
    def to_dataframe(self) -> pd.DataFrame:
        """_summary_
        Converts patient inputs to single-row DataFrame.
        Column names must match training data exactly.
        Returns:
            pd.DataFrame: _description_
        """
        return pd.DataFrame([{
            'HighBP'              : self.HighBP,              
            'HighChol'            : self.HighChol,            
            'BMI'                 : self.BMI,                 
            'Smoker'              : self.Smoker,              
            'Stroke'              : self.Stroke,              
            'HeartDiseaseorAttack': self.HeartDiseaseorAttack,
            'PhysActivity'        : self.PhysActivity,        
            'Fruits'              : self.Fruits,              
            'Veggies'             : self.Veggies,             
            'HvyAlcoholConsump'   : self.HvyAlcoholConsump,   
            'GenHlth'             : self.GenHlth,             
            'MentHlth'            : self.MentHlth,            
            'PhysHlth'            : self.PhysHlth,            
            'DiffWalk'            : self.DiffWalk,            
            'Age'                 : self.Age,                 
            'Education'           : self.Education,           
            'Income'              : self.Income,    
            }])
        
# --- Prediction pipeline -----------------------------------
class PredictPipeline:
    def __init__(self):
        self.model_path        = 'artifacts/_model/best_model_xgb.pkl'
        self.preprocessor_path = 'artifacts/_model/preprocessor.pkl'
        self._model            = None
        self._preprocessor     = None
        
    def _load_artifacts(self):
        """
        Loads model and preprocessor (lazy loading).
        """
        if self._model is None:
            logger.info("Loading model and preprocessor...")
            self._model        = joblib.load(self.model_path)
            self._preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Artifacts loaded successfully")

    def predict(self, patient: PatientData) -> dict:
        """
        Takes a PatientData object.
        Returns prediction dict with risk level and probability.

        Args:
            patient (PatientData): _description_

        Returns:
            dict: _description_
        """
        try:
            self._load_artifacts()
            
            # -- step 1: Convert to DataFrame --------------------------
            df = patient.to_dataframe()
            logger.info(f"Input received: {df.to_dict('records')[0]}")
            
            # -- Step 2: Feature engineering ----------------------------
            df = engineer_features(df)
            logger.info(f"After engineering: {df.shape[1]} features")
            
            # -- Step 3: Scale -------------------------------------------
            df_scaled = self._preprocessor.transform(df)
            
            # -- Step 4: Predict -----------------------------------------
            prediction = self._model.predict(df_scaled)[0]
            probability = self._model.predict_proba(df_scaled)[0][1]
            
            # -- Step 5: Risk level --------------------------------------
            if probability < 0.30:
                risk_level = 'Low'
                risk_color = 'Green'
            elif probability < 0.60:
                risk_level = 'Moderate'
                risk_color = 'orange'
            else:
                risk_level = 'High'
                risk_color = 'red'
            
            result = {
                'prediction'  : int(prediction),
                'probability' : round(float(probability) * 100, 1),
                'risk_level'  : risk_level,
                'risk_color'  : risk_color,
                'label'       : 'Diabetes Risk Detected' if prediction == 1 else 'Low Diabetes Risk'
            }
            
            logger.info(
                f"Prediction : {result['label']} | "
                f"Probability: {result['probability']}% | "
                f"Risk: {result['risk_level']}"
            )
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":

    # Test Patient 1 — high risk profile
    high_risk_patient = PatientData(
        HighBP=1,               # has high BP
        HighChol=1,             # has high cholesterol
        BMI=35.0,               # obese
        Smoker=1,               # smoker
        Stroke=0,
        HeartDiseaseorAttack=1, # history of heart disease
        PhysActivity=0,         # no exercise
        Fruits=0,               # no fruits
        Veggies=0,              # no vegetables
        HvyAlcoholConsump=0,
        GenHlth=4,              # fair health
        MentHlth=10,            # 10 bad mental health days
        PhysHlth=15,            # 15 bad physical health days
        DiffWalk=1,             # difficulty walking
        Age=10,                 # 65-69 years
        Education=3,            # some high school
        Income=2                # low income
    )

    # Test Patient 2 — low risk profile
    low_risk_patient = PatientData(
        HighBP=0,
        HighChol=0,
        BMI=22.0,               # normal weight
        Smoker=0,
        Stroke=0,
        HeartDiseaseorAttack=0,
        PhysActivity=1,         # exercises regularly
        Fruits=1,               # eats fruits
        Veggies=1,              # eats vegetables
        HvyAlcoholConsump=0,
        GenHlth=1,              # excellent health
        MentHlth=0,
        PhysHlth=0,
        DiffWalk=0,
        Age=3,                  # 30-34 years
        Education=6,            # college graduate
        Income=8                # high income
    )

    pipeline = PredictPipeline()

    print("\n" + "=" * 50)
    print(" TEST 1 — HIGH RISK PATIENT")
    print("=" * 50)
    result1 = pipeline.predict(high_risk_patient)
    for k, v in result1.items():
        print(f"  {k:<15}: {v}")

    print("\n" + "=" * 50)
    print(" TEST 2 — LOW RISK PATIENT")
    print("=" * 50)
    result2 = pipeline.predict(low_risk_patient)
    for k, v in result2.items():
        print(f"  {k:<15}: {v}")