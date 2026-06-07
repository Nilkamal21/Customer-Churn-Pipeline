import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="A lightweight production REST endpoint using Pydantic data shielding.",
    version="1.0.0"
)

# 2. Define the path where our trained model binary lives
# Determine the absolute base directory of the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
# 3. Load the model into memory upon server startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("--- Production model binary successfully loaded into RAM ---")
else:
    raise FileNotFoundError(f"Critical Error: Trained model weights file missing at {MODEL_PATH}. Please run train.py first.")

# 4. Define Data Validation Shield using Pydantic
class CustomerDataInput(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_credit_card: int
    is_active_member: int
    estimated_salary: float

# 5. Basic Server Health-Check Endpoint
@app.get("/")
def read_root():
    return {"status": "healthy", "service": "customer-churn-classifier"}

# 6. Prediction Engine Inference Endpoint
@app.post("/predict")
def predict_churn(input_data: CustomerDataInput):
    try:
        # Convert the Pydantic payload directly into a Pandas DataFrame row
        data_dict = input_data.model_dump()
        input_df = pd.DataFrame([data_dict])
        
        # Execute model classification prediction (0 = Stay, 1 = Churn)
        prediction = model.predict(input_df)[0]
        
        # Extract confidence score probabilities
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])

        return {
            "churn_prediction": int(prediction),
            "status_verdict": "High Risk of Leaving" if prediction == 1 else "Loyal Customer Risk Low",
            "confidence_score": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Crash: {str(e)}")