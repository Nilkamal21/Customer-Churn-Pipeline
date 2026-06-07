import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path for local module resolution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_yaml, setup_logging
from src.data_validation import DataValidation
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.model_evaluation import ModelEvaluation
import logging

class DataIngestion:
    def __init__(self, config: dict):
        self.config = config

    def initiate_data_ingestion(self) -> tuple:
        """Loads raw data, splits it into train and test datasets, and saves them to artifacts."""
        logger = logging.getLogger("MLPipeline")
        try:
            logger.info("Initializing data ingestion...")
            
            raw_data_path = self.config["data"]["raw_data_path"]
            if not os.path.exists(raw_data_path):
                raise FileNotFoundError(f"Raw dataset file not found at {raw_data_path}")
                
            # Read dataset
            df = pd.read_csv(raw_data_path)
            logger.info(f"Loaded raw dataset from {raw_data_path} with shape {df.shape}")
            
            # Split data
            test_size = self.config["data"]["test_size"]
            random_state = self.config["data"]["random_state"]
            
            logger.info(f"Splitting dataset with test_size={test_size} and random_state={random_state}")
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
            
            # Save files
            train_data_path = self.config["data"]["train_data_path"]
            test_data_path = self.config["data"]["test_data_path"]
            
            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
            
            train_set.to_csv(train_data_path, index=False)
            test_set.to_csv(test_data_path, index=False)
            
            logger.info(f"Ingested files saved: Train -> {train_data_path}, Test -> {test_data_path}")
            return train_data_path, test_data_path
            
        except Exception as e:
            logger.error(f"Error occurred during data ingestion: {e}")
            raise e

if __name__ == "__main__":
    # Orchestrate the entire ML Pipeline
    config = load_yaml("config.yaml")
    logger = setup_logging(config["logs_dir"])
    
    logger.info("==========================================")
    logger.info("Starting Customer Churn ML Training Pipeline")
    logger.info("==========================================")
    
    try:
        # Step 1: Data Validation
        validator = DataValidation(config)
        is_valid = validator.validate_all_columns()
        if not is_valid:
            logger.critical("Pipeline halted due to failed data validation.")
            exit(1)
            
        # Step 2: Data Ingestion
        ingestion = DataIngestion(config)
        train_path, test_path = ingestion.initiate_data_ingestion()
        
        # Step 3: Data Transformation
        transformation = DataTransformation(config)
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)
        
        # Step 4: Model Training
        trainer = ModelTrainer(config)
        model_path = trainer.initiate_model_trainer(X_train, y_train)
        
        # Step 5: Model Evaluation
        evaluation = ModelEvaluation(config)
        metrics = evaluation.evaluate_model(model_path, X_test, y_test)
        
        logger.info("==========================================")
        logger.info("ML Training Pipeline Completed Successfully!")
        logger.info("==========================================")
        
    except Exception as e:
        logger.critical(f"ML Pipeline execution failed: {e}", exc_info=True)
        exit(1)
