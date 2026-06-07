import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.predict_pipeline import PredictPipeline

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="A modular production REST endpoint using Pydantic data validation and clean project architecture.",
    version="1.0.0"
)

# 2. Define Data Validation Shield using Pydantic
class CustomerDataInput(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_credit_card: int
    is_active_member: int
    estimated_salary: float

# 3. Initialize prediction pipeline
try:
    predict_pipeline = PredictPipeline(config_path="config.yaml")
    print("--- Prediction Engine Initialized Successfully ---")
except Exception as e:
    print(f"Warning: Prediction Engine initialization failed: {e}. Make sure a model is trained first.")

# 4. Basic Server Health-Check Endpoint
@app.get("/")
def read_root():
    return {"status": "healthy", "service": "customer-churn-classifier"}

# 5. Prediction Engine Inference Endpoint
@app.post("/predict")
def predict_churn(input_data: CustomerDataInput):
    try:
        # Convert the Pydantic payload directly into a Pandas DataFrame row
        data_dict = input_data.model_dump()
        input_df = pd.DataFrame([data_dict])
        
        # Execute model classification prediction (0 = Stay, 1 = Churn)
        prediction, probabilities = predict_pipeline.predict(input_df)
        confidence = float(probabilities[prediction])

        return {
            "churn_prediction": int(prediction),
            "status_verdict": "High Risk of Leaving" if prediction == 1 else "Loyal Customer Risk Low",
            "confidence_score": round(confidence, 4)
        }

    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Crash: {str(e)}")
