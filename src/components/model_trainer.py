# model_trainer.py 

import os
import sys
import json

from src.logger import logger
from src.exception import CustomException

import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score,
    f1_score, precision_score,
    accuracy_score, classification_report
)
import joblib

# --- Configuration --------------------------------------
@dataclass
class ModelTrainerConfig:
    model_path   : str = os.path.join('artifacts', '_model', 'best_model_xgb.pkl')
    metadata_path: str = os.path.join('artifacts', '_model', 'model_metadata.json')
    
# --- Main components for model training ------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        
    def _evaluate(self, model, X_test, y_test) -> dict:
        """
        Evaluates model on test set.
        Returns dict of all metrics.

        Args:
            model (_type_): _description_
            X_test (_type_): _description_
            y_test (_type_): _description_

        Returns:
            dict: _description_
        """
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        return {
            'roc_auc'  : round(roc_auc_score(y_test, y_prob), 4),
            'recall'   : round(recall_score(y_test, y_pred), 4),
            'F1'       : round(f1_score(y_test, y_pred), 4), 
            'precision': round(precision_score(y_test, y_pred), 4),
            'accuracy' : round(accuracy_score(y_test, y_pred), 4)
        }
        
    def initiate_model_training(
        self,
        X_train_smote : np.ndarray,   # SMOTE balanced (for LR)
        y_train_smote : pd.Series,    # SMOTE balanced (for LR)
        X_train_scaled: np.ndarray,   # Original scaled (for XGB)
        y_train_orig  : pd.Series,    # Original imbalanced (for XGB)
        X_test        : np.ndarray,
        y_test        : pd.Series,  
        ):
        """
        Trains XGBoost with best params from GridSearchCV.
        Evaluates on test set.
        Saves model + metadata to artifacts/_model/

        Args:
            X_train (np.ndarray): _description_
            y_train (pd.Series): _description_
            X_test (np.ndarray): _description_
            y_test (pd.Series): _description_
        """
        
        logger.info("Model training started")
        
        try:
            # -- Calculate scale_pos_weight ----------------------
            neg = (y_train_orig  == 0).sum()
            pos = (y_train_orig  == 1).sum()
            scale_pos_weight = round(neg/ pos, 4)
            
            logger.info(f"scale_pos_weight = {neg:,} / {pos:,} = {scale_pos_weight}")
            
            # # -- Parameter grid -----------------------------------
            # param_grid = {
            #     'n_estimators'    : [100, 200, 300],
            #     'max_depth'       : [3, 4, 5, 6],
            #     'learning_rate'   : [0.01, 0.05, 0.1],
            #     'subsample'       : [0.7, 0.8, 1.0],
            #     'scale_pos_weight': [scale_pos_weight]
            # }

            # total = 3 * 4 * 3 * 3
            # logger.info(
            #     f"GridSearchCV: {total} combinations "
            #     f"x 5 folds = {total*5} fits"
            # )
            
            # # -- GriedSearcjCV -----------------------------------
            # cv = StratifiedKFold(
            #     n_splits=5, shuffle=True, random_state=42
            # )
            # grid_search = GridSearchCV(
            #     estimator=XGBClassifier(
            #     eval_metric='logloss',
            #     verbosity=0,
            #     randome_state=42  
            #     ),
            #     param_grid=param_grid,
            #     scoring='roc_auc',
            #     cv=cv,
            #     n_jobs=-1,
            #     verbose=0,
            #     refit=True
            # )
            
            # logger.info(
            #     f"Running GridSearchCV on {X_train_scaled.shape[0]:,} "
            #     f"samples (original + scale_pos_weight)..."
            # )
            # grid_search.fit(X_train_scaled, y_train_orig)
            
            # best_params = grid_search.best_params_
            # best_cv_auc = grid_search.best_score_
            # model = grid_search.best_estimator_
            
            # logger.info(f"Best params: {best_params}")
            # logger.info(f"Best CV AUC: {best_cv_auc:.4f}")
            
            # # -- Evaluate on test set ----------------------------
            # metrics = self._evaluate(model, X_test, y_test)
            # logger.info(f"Test metrics: {metrics}")
            
            
             
            # # -- Best params from GridSearchCV --------------------
            # ── Best params — tuned once in notebook 2 ────────
            # Source: GridSearchCV 5-fold CV on CDC BRFSS 2015
            # CV AUC: 0.8285 | Test AUC: 0.8242
            best_params = {
                'learning_rate'     : 0.05,
                'max_depth'         : 4,
                'n_estimators'      : 200,
                'subsample'         : 0.7,
                'scale_pos_weight'  : scale_pos_weight,
                'eval_metric'       : 'logloss',
                'verbosity'         : 0,
                'random_state'      : 42  
            }
            logger.info(f"Best params for XGBClassifier from GridSearchCV: {best_params}")
            
            # -- Train XGBClassifier model ---------------------------
            logger.info(
                    f"Training XGBoost on {X_train_scaled.shape[0]:,}"
                    f"samples x {X_train_scaled.shape[1]} features."
                    "(original imbalanced + scale_pos_weight)"
                )

            model = XGBClassifier(**best_params)
            model.fit(X_train_scaled, y_train_orig)
            logger.info("XGBoost training complete")
            
            # -- Evaluate --------------------------------------------
            metrics = self._evaluate(model, X_test, y_test)
            logger.info(f"test metrics after evaluate: {metrics}")
            
            
            
            # -- Validate minimum performance ------------------------
            if metrics['roc_auc'] < 0.80:
                logger.warning(
                    f"ROC_AUC {metrics['roc_auc']} below the threshold 0.80 - check pipeline"
                )
            else:
                logger.info(
                    f"ROC-AUC {metrics['roc_auc']} meets production threshold"
                )
                
            # -- Save Model ------------------------------------------
            os.makedirs(
                os.path.dirname(self.config.model_path),
                exist_ok=True
            )
            
            joblib.dump(model, self.config.model_path)
            logger.info(
                f"XGBoost model saved : {self.config.model_path}"
            )
            
            # -- Save metadata -------------------------------------------
            metadata = {
                'model_name'   : 'XGBoost (GridSearchCV tuned)',
                'saved_at'     : datetime.now().strftime(
                                '%Y-%m-%d %H:%M'),
                'best_params'  : best_params,
                'performance'  : metrics,
                'dataset_info' : {
                    'train_samples'   : int(X_train_scaled.shape[0]),
                    'test_samples'    : int(X_test.shape[0]),
                    'n_features'      : int(X_train_scaled.shape[1]),
                    'smote_applied'   : False,
                    'strategy'        : 'scale_pos_weight',
                    'scale_pos_weight': scale_pos_weight
                },
                'thresholds': {
                    'min_roc_auc': 0.80,
                    'passed'     : metrics['roc_auc'] >= 0.80
                }
            
            }
            with open(self.config.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(
                f"Metadata saved: {self.config.metadata_path}"
            )

            logger.info("Model training complete")
            return self.config.model_path, metrics
            
        except Exception as e:
            raise CustomException(e, sys)
        
# --- Quick test ------------------------------------------------------
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # step 1: Data Ingestion----
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    # step 2: Data Transformation ----
    transformation = DataTransformation()
    (X_train_smote, y_train_smote,
     X_train_scaled, y_train_orig,
     X_test, y_test,
     preprocessor_path) = transformation.initiate_data_transformation(train_path, test_path)
    
    # step 3: Model train using prepared data ----
    trainer = ModelTrainer()
    model_path, metrics = trainer.initiate_model_training( 
        X_train_smote,  y_train_smote,
        X_train_scaled, y_train_orig,
        X_test,         y_test
    ) 
    
    print(f"\n Model saved : {model_path}")
    print("\n Final Metrics:")
    for k, v in metrics.items():
        print(f" {k:<12}: {v:.4f}")