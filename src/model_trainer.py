import os
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_object
import logging

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config

    def initiate_model_trainer(self, X_train, y_train) -> str:
        """Trains the Random Forest model and saves it to the artifacts directory."""
        try:
            logger = logging.getLogger("MLPipeline")
            logger.info("Initializing model training phase...")
            
            # Read model config parameters
            n_estimators = self.config["model"]["n_estimators"]
            random_state = self.config["model"]["random_state"]
            model_path = self.config["model"]["model_path"]
            
            logger.info(f"Training Random Forest Classifier with n_estimators={n_estimators}")
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)
            
            logger.info(f"Saving trained model to {model_path}...")
            save_object(file_path=model_path, obj=model)
            
            logger.info("Model training completed successfully and saved.")
            return model_path
            
        except Exception as e:
            logging.getLogger("MLPipeline").error(f"Error during model training: {e}")
            raise e
