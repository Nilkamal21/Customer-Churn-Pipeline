import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import load_object
import logging

class ModelEvaluation:
    def __init__(self, config: dict):
        self.config = config

    def evaluate_model(self, model_path: str, X_test, y_test):
        """Loads the model, predicts on test data, logs runs/metrics/models to MLflow."""
        logger = logging.getLogger("MLPipeline")
        try:
            logger.info("Initializing model evaluation...")
            model = load_object(model_path)
            
            # Predict
            predictions = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            logger.info(f"Model Metrics: Accuracy={accuracy*100:.2f}%, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            # MLflow configuration
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
            
            run_name = self.config["mlflow"]["run_name"]
            with mlflow.start_run(run_name=run_name):
                # Log model parameters
                n_estimators = self.config["model"]["n_estimators"]
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("test_size", self.config["data"]["test_size"])
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
                # Log model
                registered_model_name = self.config["model"]["registered_model_name"]
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=registered_model_name
                )
                
                logger.info(f"MLflow Run successfully completed. Model registered as '{registered_model_name}'.")
                
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation or MLflow logging: {e}")
            raise e
