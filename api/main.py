from fastapi import FastAPI
from pydantic import BaseModel
from model import retrain_model  # Import retraining function

app = FastAPI()

# Define request body for retraining
class RetrainRequest(BaseModel):
    data_url: str  # URL or path to new data (can be customized)

@app.post("/retrain")
async def retrain(request: RetrainRequest):
    # Example: Call a function to retrain the model
    retrain_model(request.data_url)  # Pass in the new data URL or path
    return {"message": "Model retraining triggered!"}

