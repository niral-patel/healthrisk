# src/utils.py
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single source of truth for feature engineering.
    Used by both DataTransformation and PredictPipeline.
    Any change here applies to both automatically.
    """
    df = df.copy()
    df['BMI'] = df['BMI'].clip(lower=13.5, upper=41.5)
    df['HealthScore']    = df['MentHlth'] + df['PhysHlth']
    df['SocioScore']     = df['Income']   + df['Education']
    df['LifestyleScore'] = (df['PhysActivity'] +
                            df['Fruits'] + df['Veggies'])
    risk_cols = ['HighBP','HighChol','Smoker',
                 'Stroke','HeartDiseaseorAttack','HvyAlcoholConsump']
    df['RiskFactorCount'] = df[risk_cols].sum(axis=1)

    def bmi_cat(bmi):
        if bmi < 18.5: return 0
        elif bmi < 25: return 1
        elif bmi < 30: return 2
        else:          return 3
    df['BMICategory'] = df['BMI'].apply(bmi_cat)

    def age_risk(age):
        if age <= 4:   return 0
        elif age <= 8: return 1
        else:          return 2
    df['AgeGroup'] = df['Age'].apply(age_risk)

    return df