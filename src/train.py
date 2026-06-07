import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

def run_training_pipeline():
    # 1. Force MLflow to use a local SQLite database for clean tracking on Windows
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Force MLflow to use an absolute SQLite path
    db_path = os.path.join(BASE_DIR, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Customer_Churn_Tracking")
    
    with mlflow.start_run(run_name="Random_Forest_Production_Run"):
        print("--- Phase 1: Ingesting Customer Data ---")
        # Direct absolute layout pointing to your root data folder
        data_path = os.path.join(BASE_DIR, "data", "customer_churn.csv")
        df = pd.read_csv(data_path)
        print(f"Loaded dataset containing {df.shape[0]} customer records.")

        print("\n--- Phase 2: Preprocessing Features ---")
        X = df.drop(columns=["churn"])
        y = df["churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\n--- Phase 3: Training Classifier ---")
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Log Hyperparameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)

        print("\n--- Phase 4: Evaluating Metrics ---")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Baseline Accuracy Score: {accuracy * 100:.2f}%")
        
        # Log Metric to MLflow
        mlflow.log_metric("accuracy", accuracy)

        print("\n--- Phase 5: Exporting Production Artifacts ---")
        os.makedirs("models", exist_ok=True)
        model_output_path = os.path.join("models", "churn_model.pkl")
        joblib.dump(model, model_output_path)
        
        # Log Model directly into the MLflow Registry
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="Churn_RandomForest_Model")
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_training_pipeline()