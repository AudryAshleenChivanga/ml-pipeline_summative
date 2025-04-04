from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from PIL import Image
import numpy as np
import io
import os
import uuid
import logging
import traceback
from typing import List
from pymongo import MongoClient
from gridfs import GridFS
import tensorflow as tf
import glob
from fastapi.middleware.cors import CORSMiddleware
import traceback

# ------------------------- App Setup -------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


# ------------------------- Constants -------------------------

MODEL_DIR = "models"
BASE_MODEL_NAME = "CNN_Model_Updated"
MONGODB_URI = "MONGODB_URI" 
DB_NAME = "glaucoma_db"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# ------------------------- MongoDB Setup -------------------------

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
fs = GridFS(db)

# ------------------------- Helper Functions -------------------------

def get_latest_model():
    """Get the latest model from the models directory"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_files = glob.glob(os.path.join(MODEL_DIR, f"{BASE_MODEL_NAME}*.h5"))
    
    if not model_files:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), 
                   padding='same', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer=Adam(),
                     loss='categorical_crossentropy',
                     metrics=['accuracy', 'Precision', 'Recall'])
        
        model_path = os.path.join(MODEL_DIR, f"{BASE_MODEL_NAME}1.h5")
        model.save(model_path)
        return model, model_path
    else:
        latest_model = max(model_files, key=lambda x: int(x.split(BASE_MODEL_NAME)[1].split('.')[0]))
        return load_model(latest_model), latest_model

def save_new_model_version(model):
    """Save the model with an incremented version number"""
    existing_versions = []
    for f in os.listdir(MODEL_DIR):
        if f.startswith(BASE_MODEL_NAME) and f.endswith('.h5'):
            try:
                version = int(f[len(BASE_MODEL_NAME):-3])
                existing_versions.append(version)
            except:
                continue
    
    next_version = max(existing_versions) + 1 if existing_versions else 1
    new_model_path = os.path.join(MODEL_DIR, f"{BASE_MODEL_NAME}{next_version}.h5")
    model.save(new_model_path)
    return new_model_path

def preprocess_image_bytes(image_bytes):
    """Process image for prediction"""
    img = Image.open(io.BytesIO(image_bytes)).resize(IMAGE_SIZE).convert("RGB")
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_for_storage(image_bytes):
    """Process image for storage in MongoDB"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", quality=95)
        return buffer.getvalue()
    except Exception as e:
        logging.error(f"Image processing failed: {str(e)}")
        raise ValueError(f"Could not process image: {str(e)}")

def fetch_dataset_from_mongo(split):
    """Fetch images and labels from MongoDB"""
    images, labels = [], []
    
    for record in db.images.find({"split": split}):
        try:
            file_data = fs.get(record["file_id"]).read()
            img = Image.open(io.BytesIO(file_data)).resize(IMAGE_SIZE).convert("RGB")
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(1 if record["diagnosis"] == "glaucoma" else 0)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            continue
            
    return np.array(images), to_categorical(labels, 2)

# ------------------------- API Endpoints -------------------------

@app.get("/")
def health_check():
    return {
        "status": "Healthy",
        "current_model": os.path.basename(current_model_path) if 'current_model_path' in globals() else "N/A"
    }

@app.post("/predict/")
async def predict_glaucoma(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        processed_image = preprocess_image_bytes(image_bytes)
        prediction = model.predict(processed_image)
        class_idx = int(np.argmax(prediction))
        class_name = "Glaucoma" if class_idx == 1 else "Normal"
        probability = float(prediction[0][class_idx])
        return {
            "prediction": class_name,
            "probability": probability,
            "model_used": os.path.basename(current_model_path)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/upload/")
async def upload_image(
    image: UploadFile = File(...),
    diagnosis: str = Form(...),
    split: str = Form("train")
):
    try:
        diagnosis = diagnosis.lower()
        split = split.lower()

        if diagnosis not in ["glaucoma", "normal"]:
            raise ValueError("Diagnosis must be 'glaucoma' or 'normal'")
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        image_bytes = await image.read()
        processed_bytes = preprocess_for_storage(image_bytes)
        
        file_id = fs.put(processed_bytes, filename=f"{uuid.uuid4()}.png")
        
        db.images.insert_one({
            "file_id": file_id,
            "original_filename": image.filename,
            "diagnosis": diagnosis,
            "split": split
        })

        return {"status": "success", "message": "Image uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

@app.post("/upload_folder/")
async def upload_folder(
    files: List[UploadFile] = File(...),
    diagnosis: str = Form(...),
    split: str = Form("train")
):
    try:
        diagnosis = diagnosis.lower()
        split = split.lower()

        if diagnosis not in ["glaucoma", "normal"]:
            raise ValueError("Diagnosis must be 'glaucoma' or 'normal'")
        if split not in ["train", "val", "test"]:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        uploaded_files = []
        for file in files:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                image_bytes = await file.read()
                processed_bytes = preprocess_for_storage(image_bytes)
                
                file_id = fs.put(processed_bytes, filename=f"{uuid.uuid4()}.png")
                
                db.images.insert_one({
                    "file_id": file_id,
                    "original_filename": file.filename,
                    "diagnosis": diagnosis,
                    "split": split
                })
                uploaded_files.append(file.filename)
            except Exception as e:
                logging.error(f"Failed to process {file.filename}: {str(e)}")
                continue

        return {
            "status": "success",
            "message": f"Uploaded {len(uploaded_files)} images",
            "uploaded_files": uploaded_files
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")

@app.post("/retrain/")
async def retrain_model():
    global model, current_model_path
    
    try:
        # Load data
        X_train, y_train = fetch_dataset_from_mongo("train")
        X_val, y_val = fetch_dataset_from_mongo("val")
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient training data. Please upload more images.")

        # Create new model instance
        model_config = model.get_config()
        new_model = Sequential.from_config(model_config)
        new_model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        new_model.set_weights(model.get_weights())

        # Create tf.data.Dataset with augmentation
        def augment(x, y):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, 0.2)
            x = tf.image.random_contrast(x, 0.8, 1.2)
            return x, y

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1024).map(augment).batch(BATCH_SIZE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )

        # Train
        history = new_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        new_model_path = save_new_model_version(new_model)
        model = load_model(new_model_path)
        current_model_path = new_model_path

        # Metrics
        y_pred = model.predict(X_val, batch_size=BATCH_SIZE)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        metrics = {
            "accuracy": accuracy_score(y_true_classes, y_pred_classes),
            "precision": precision_score(y_true_classes, y_pred_classes),
            "recall": recall_score(y_true_classes, y_pred_classes),
            "f1_score": f1_score(y_true_classes, y_pred_classes),
            "classification_report": classification_report(y_true_classes, y_pred_classes, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true_classes, y_pred_classes).tolist(),
            "training_history": history.history,
            "new_model": os.path.basename(new_model_path)
        }

        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics
        }
        
    except Exception as e:
        logging.error(f"Retraining failed: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Retraining failed. Check server logs."
            }
        )

@app.get("/dataset_stats/")
async def get_dataset_stats():
    stats = {
        "total_images": db.images.count_documents({}),
        "glaucoma": {
            "train": db.images.count_documents({"diagnosis": "glaucoma", "split": "train"}),
            "val": db.images.count_documents({"diagnosis": "glaucoma", "split": "val"}),
            "test": db.images.count_documents({"diagnosis": "glaucoma", "split": "test"})
        },
        "normal": {
            "train": db.images.count_documents({"diagnosis": "normal", "split": "train"}),
            "val": db.images.count_documents({"diagnosis": "normal", "split": "val"}),
            "test": db.images.count_documents({"diagnosis": "normal", "split": "test"})
        }
    }
    return stats

# Initializing the model
model, current_model_path = get_latest_model()
