import os
import pandas as pd
import logging

class DataTransformation:
    def __init__(self, config: dict):
        self.config = config

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Reads train/test CSVs and splits them into input features and target feature."""
        try:
            logger = logging.getLogger("MLPipeline")
            logger.info("Starting data transformation / feature splitting...")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_col = self.config["data"]["target_column"]
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            logger.info(f"Feature split completed. Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.getLogger("MLPipeline").error(f"Error during data transformation: {e}")
            raise e
