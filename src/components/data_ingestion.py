# data_ingestion.py: Component for ingesting and preprocessing data

import os
import sys

import pandas as pd
import numpy as np

from src.logger import logger
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# -- Configuration Class ---------------------------------
@dataclass
class DataIngestionConfig:
    """_summary_
    data_path: str = os.path.join('artifacts', 'data.csv')
    
    Stores all file paths used by DataIngestion.
    dataclass automatically creates __init__ from fields.
    
    """
    raw_data_path  : str = os.path.join('artifacts', '_data', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', '_data', 'train.csv')
    test_data_path : str = os.path.join('artifacts', '_data', 'test.csv')
    
    
# --- Main Component ---------------------------------
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Reads raw data, splits into train/test, and saves to disk.
        saves both to artifacts/, returns their paths.
        Returns:
            Tuple[str, str]: Paths to train and test data files.
        Raises:
            CustomException: If any error occurs during data ingestion.
        """
        logger.info("Starting data ingestion process.")
        
        try:
            # -- step 1: Load dataset ---------------------------------------
            data_path = os.path.join(
                'notebook', 'data',
                'diabetes_binary_health_indicators_BRFSS2015.csv'
            )
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully from {data_path}."
                           f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns.")

            # -- step 2: Validate datasets --------------------------------------
            if df.empty:
                raise CustomException("Loaded dataset is empty.", sys)
            
            if df.shape[0] == 0:
                raise CustomException("No data found in the dataset.", sys)
            
            if 'Diabetes_binary' not in df.columns:
                raise CustomException("Target column 'Diabetes_binary' is missing.", sys)
            
            missing_pct = df.isnull().sum().sum() / df.size * 100
            logger.info(f"Percentage of missing values in the dataset: {missing_pct:.2f}%")
            
            # -- step 3: Save raw copy --------------------------------------
            os.makedirs(
                os.path.dirname(self.config.raw_data_path),
                exist_ok=True
            ) 
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw dataset saved to {self.config.raw_data_path}.")
            
            #-- step 4: Split into train/test --------------------------------------
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['Diabetes_binary']
            )
            logger.info(
                f"Train dataset shape: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns."
                f"Test dataset shape: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns."
            )

            #-- step 5: Save train/test --------------------------------------
            train_df.to_csv(
                self.config.train_data_path, index=False
            )
            test_df.to_csv(
                self.config.test_data_path, index=False
            )

            logger.info(
                f"Train dataset saved to {self.config.train_data_path}."
            )
            logger.info(
                f"Test dataset saved to {self.config.test_data_path}."
            )
            logger.info("Data ingestion process completed successfully.")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
            
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)
        
        
# -- Quick test ---------------------------------
if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data ingestion test completed. Train path: {train_path}, Test path: {test_path}")
    
    except Exception as e:
        logger.error(f"Data ingestion test failed: {e}")