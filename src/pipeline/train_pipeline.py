# train_pipeline.py

import sys

from src.logger import logger
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.data_ingestion      = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer       = ModelTrainer()
        
    
    def run(self):
        """
        Runs the full training pipeline:
        DataIngestion -> DataTransformation -> ModelTrainer
        """
        try:
            logger.info("=" * 55)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 55)
            
            # -- Stage 1: Data Ingestion ---------------------------
            logger.info("Stage 1: Data Ingestion")
            train_path, test_path = (self.data_ingestion.initiate_data_ingestion())
            
            # -- Stage 2: Data Transformation ------------------------
            logger.info("Stage 2: Data Transformation")
            (X_train_smote, y_train_smote, 
             X_train_scaled, y_train_orig, 
             X_test, y_test, 
             preprocessor_path) = (
                 self.data_transformation.initiate_data_transformation(
                     train_path, 
                     test_path
                     )
                 )
            
            # -- Stage 3: Model Training -----------------------------
            logger.info("Stage 3: Model Training")
            model_path, metrics = (
                self.model_trainer.initiate_model_training(
                    X_train_smote, y_train_smote,
                    X_train_scaled, y_train_orig,
                    X_test, y_test
                )
            )
            
            logger.info("=" * 55)
            logger.info("TRAINING PIPELINE COMPLETE")
            logger.info(f"Model     : {model_path}")
            logger.info(f"ROC-AUC   : {metrics['roc_auc']}")
            logger.info(f"Recall    : {metrics['recall']}")
            logger.info(f"F1 Score  : {metrics['F1']}")
            logger.info("=" * 55)
            
            return model_path, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    pipeline = TrainPipeline()
    model_path, metrics = pipeline.run()

    print(f"\n Pipeline complete!")
    print(f" Model    : {model_path}")
    print(f" ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f" Recall   : {metrics['recall']:.4f}")
    print(f" F1 Score : {metrics['F1']:.4f}")