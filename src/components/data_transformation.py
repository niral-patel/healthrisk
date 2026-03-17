# data_transformation.py: Component for transforming data (e.g., cleaning, feature engineering, scaling)

import os
import sys

import pandas as pd
import joblib
from dataclasses import dataclass

from src.logger import logger
from src.exception import CustomException
from src.utils import engineer_features

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE



# -- Configuration Class ---------------------------------
@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation paths.
    """
    
    preprocessor_path: str = os.path.join(
        'artifacts', '_model', 'preprocessor.pkl')
    
    
    
# --- Main Component ---------------------------------
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        
    # --- Build preprocessor (mirrors notebook) -------------------------------
    def _build_preprocessor(self) -> Pipeline:
        """
        Builds a preprocessing (StandardScaler pipeline) pipeline for the dataset.
        Fit on train only — transform both train and test.
        Args:
            df (pd.DataFrame): _description_

        Returns:
            Pipeline: _description_
        """
        preprocessor = Pipeline(steps=[
            ('scaler', StandardScaler())
            ])
        return preprocessor
    
    # --- Main method to run transformation -------------------------------
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str
        ):
        """
        Main method to run data transformation.
        Full transformation pipeline:
        1. Load train/test CSVs
        2. Engineer features
        3. Separate X and y
        4. Fit scaler on train, transform both
        5. Apply SMOTE on train only
        6. Save preprocessor
        7. Return arrays + preprocessor path
        Args:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the testing data CSV file.
        Raises:
            CustomException: If any error occurs during data transformation.
        """
        logger.info("Starting data transformation process.")
    
        try:
            # -- Load datasets --------------------------------------
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Train dataset loaded from {train_path}. Shape: {train_df.shape}")
            logger.info(f"Test dataset loaded from {test_path}. Shape: {test_df.shape}")
            
            # -- Feature engineering --------------------------------
            train_df = engineer_features(train_df)
            test_df  = engineer_features(test_df)
            logger.info(
                f"After engineering -"
                f"train {train_df.shape} | "
                f"test {test_df.shape}"
            )
            
            # -- Separate X and y -----------------------------------
            target  = 'Diabetes_binary'
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]
            X_test  = test_df.drop(columns=[target])
            y_test  = test_df[target]
            
            logger.info(
                f"X_train: {X_train.shape} |"
                f"X_test:  {X_test.shape}"
            )
            
            logger.info(
                f"Class balance — "
                f"train: {y_train.value_counts().to_dict()} | "
                f"test: {y_test.value_counts().to_dict()}"
            )
            
            # -- Scale --------------------------------------------
            preprocessor   = self._build_preprocessor()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled   = preprocessor.transform(X_test)
            
            logger.info(
                "StandardScaler fitted on train,"
                "applied to both train and test"
            )
            
            # -- SMOT on train only --------------------------------
            smote = SMOTE(random_state=42)
            
            X_train_smote, y_train_smote = smote.fit_resample(
                X_train_scaled, y_train
            )
            
            logger.info(
                f"SMOTE applied — "
                f"before: {X_train_scaled.shape[0]:,} | "
                f"after: {X_train_smote.shape[0]:,}"
            )
            logger.info(
                f"After SMOTE balance: "
                f"{pd.Series(y_train_smote).value_counts().to_dict()}"
            )
            
            logger.info("Data Transformation complete")
            
            # -- Save preprocessor ---------------------------------
            os.makedirs(
                os.path.dirname(self.config.preprocessor_path),
                exist_ok= True
            )
            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info(f'Preprocessor saved: {self.config.preprocessor_path}')
            
            return (
                X_train_smote,
                y_train_smote,
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                self.config.preprocessor_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        

# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    # Step 1: Run ingestion first
    ingestion  = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Run transformation
    transformation = DataTransformation()
    (X_train_smote, y_train_smote,
     X_train_scaled, y_train_orig,
     X_test, y_test,
     preprocessor_path) = transformation.initiate_data_transformation(
        train_path, test_path
    )

    print(f"\nX_train_smote  :{X_train_smote.shape}")
    print(f"y_train_smote    : {y_train_smote.shape}")
    print(f"X_train_scaled   : {X_train_scaled.shape}")
    print(f"X_test           : {X_test.shape}")
    print(f"y_test           : {y_test.shape}")
    print(f"Preprocessor     : {preprocessor_path}")
    print(f"\nClass balance after SMOTE:")
    print(f"  {pd.Series(y_train_smote).value_counts().to_dict()}")