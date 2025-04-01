import os
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import shutil
from io import BytesIO

# Create FastAPI application
app = FastAPI()

# Load the model once at startup
MODEL_PATH = 'models/CNN_Model_Updated.h5'
model = load_model(MODEL_PATH)

# API router for prediction and retraining
router = APIRouter()

# Prediction endpoint
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        img = Image.open(BytesIO(image_data)).resize((128, 128)).convert('RGB')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        prediction_class = np.argmax(prediction, axis=1)
        diagnosis = "Glaucoma" if prediction_class[0] == 1 else "Normal"

        return {"prediction": diagnosis}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Retrain endpoint (you can call your existing retrain function here)
@router.post("/retrain")
async def retrain():
    try:
        # Add your retraining logic here
        retrain_model()
        return {"message": "Retraining successful!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Retraining model function (fill in the logic you already have for retraining)
def retrain_model():
    # This is a placeholder, replace it with your existing retraining logic
    print("🔄 Retraining model...")
    # Example:
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
    # model.save("models/CNN_Model_Updated.h5")
    print("💾 Model retrained and saved successfully!")

# Register router
app.include_router(router, prefix="/api/v1")

# To run this app, use the following command:
# uvicorn main:app --reload
# Ensure you have the required packages installed: