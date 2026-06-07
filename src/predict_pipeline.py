import os
import pandas as pd
from src.utils import load_yaml, load_object

class PredictPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration to get model path
        self.config = load_yaml(config_path)
        self.model_path = self.config["model"]["model_path"]

    def predict(self, features: pd.DataFrame) -> tuple:
        """Runs prediction on features and returns (prediction, probabilities)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Make sure the training pipeline was executed.")
            
        model = load_object(self.model_path)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return prediction, probabilities
