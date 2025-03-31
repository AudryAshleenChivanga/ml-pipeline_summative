import os
from pathlib import Path
import io
from PIL import Image
import numpy as np
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import random
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Configuration
IMAGES_PER_CLASS = 30  # 60 per split (30 normal, 30 glaucoma)
IMAGE_SIZE = (128, 128, 3)
SPLITS = ['train', 'val', 'test']
MAX_RETRIES = 3
RETRY_DELAY = 2
THREADS = 5  # Number of parallel uploads
BATCH_SIZE = 32
EPOCHS = 10


def get_mongo_connection():
    """Robust MongoDB connection with retries and DNS cache"""
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    
    if not uri:
        raise ValueError("MONGODB_URI not found in .env")
    
    # Configure DNS cache
    socket.setdefaulttimeout(10)
    
    for attempt in range(MAX_RETRIES):
        try:
            client = MongoClient(
                uri,
                connectTimeoutMS=20000,
                socketTimeoutMS=60000,
                serverSelectionTimeoutMS=20000,
                retryWrites=True,
                retryReads=True
            )
            client.admin.command('ping')
            return client
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"⚠️ Connection attempt {attempt + 1} failed: {str(e)}")
            time.sleep(RETRY_DELAY)


def fetch_images_from_mongo(db, split):
    """Fetch images and labels from MongoDB"""
    images, labels = [], []
    
    cursor = db.images.find({"split": split})
    
    for doc in cursor:
        file_id = doc["file_id"]
        label = 1 if doc["diagnosis"] == "glaucoma" else 0
        
        file_data = db.fs.files.find_one({"_id": file_id})
        if not file_data:
            continue
        
        image_data = db.fs.chunks.find_one({"files_id": file_id})
        if not image_data:
            continue
        
        img = Image.open(io.BytesIO(image_data["data"])).resize(IMAGE_SIZE[:2]).convert("RGB")
        images.append(img_to_array(img) / 255.0)
        labels.append(label)
    
    return np.array(images), to_categorical(labels, 2)


def retrain_model():
    """Load, retrain, and save the CNN model"""
    client = get_mongo_connection()
    db = client[os.getenv("DB_NAME", "glaucoma_db")]
    
    print("📥 Loading images from MongoDB...")
    X_train, y_train = fetch_images_from_mongo(db, "train")
    X_val, y_val = fetch_images_from_mongo(db, "val")
    
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print("❌ No images found in MongoDB for training!")
        return
    
    print("✅ Images loaded successfully!")
    
    print("📥 Loading existing model...")
    model = load_model("models/CNN_Model_1.h5")
    
    print("🔄 Retraining model...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    print("💾 Saving updated model...")
    model.save("models/CNN_Model_Updated.h5")
    print("✅ Model retrained and saved successfully!")
    
    client.close()


if __name__ == "__main__":
    retrain_model()
