import os
import logging
import numpy as np
from pymongo import MongoClient
from gridfs import GridFS
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Constants
BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = (128, 128, 3)
MODEL_PATH = "/content/model_base.h5"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load model if exists
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
else:
    logging.error(f"Model file {MODEL_PATH} not found! Please upload it.")
    model = None

# Define the model structure if not loaded
if model is None:
    model = models.Sequential([
        layers.InputLayer(input_shape=IMAGE_SIZE),
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client["glaucoma_db"]
fs = GridFS(db)

def fetch_images_from_mongo(split):
    """
    Fetch images from MongoDB and preprocess them.
    """
    images, labels = [], []
    cursor = db.images.find({"split": split})
    
    for doc in cursor:
        try:
            file_id = doc["file_id"]
            label = 1 if doc["diagnosis"] == "glaucoma" else 0
            file_data = db.fs.chunks.find_one({"files_id": file_id})
            
            if file_data:
                img = Image.open(io.BytesIO(file_data["data"])).resize((128, 128)).convert("RGB")
                images.append(np.array(img) / 255.0)  # Normalize image
                labels.append(label)
            else:
                logging.warning(f"Image data for file_id {file_id} not found!")
        except Exception as e:
            logging.error(f"Error processing image with file_id {file_id}: {e}")

    return np.array(images), to_categorical(labels, 2)

def create_data_augmentation_pipeline(X_train):
    """
    Create a data augmentation pipeline for training images.
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    return datagen

def visualize_and_save_augmented_images(datagen, X_train, num_images=5):
    """
    Visualize and save a few augmented images.
    """
    i = 0
    for batch in datagen.flow(X_train, batch_size=num_images, save_to_dir='./',
                              save_prefix='augmented', save_format='png'):
        plt.figure(figsize=(10, 10))
        for j in range(num_images):
            plt.subplot(1, num_images, j + 1)
            plt.imshow(batch[j])
            plt.axis('off')
        plt.show()
        i += 1
        if i > 0:
            break

# Fetch data from MongoDB
X_train, y_train = fetch_images_from_mongo("train")
X_val, y_val = fetch_images_from_mongo("val")

if X_train.shape[0] == 0 or X_val.shape[0] == 0:
    logging.error("No images were loaded. Check MongoDB data integrity.")
else:
    logging.info(f"📊 Loaded {X_train.shape[0]} training images and {X_val.shape[0]} validation images")

# Augment the training data
datagen = create_data_augmentation_pipeline(X_train)
visualize_and_save_augmented_images(datagen, X_train, num_images=5)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    callbacks=[early_stopping])

# Save the trained model
model.save("CNN_Model_Updated.h5")
logging.info("✅ Model retrained and saved as CNN_Model_Updated.h5")

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on validation data.
    """
    loss, accuracy, precision, recall = model.evaluate(X_val, y_val)
    logging.info(f"Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

# Call the evaluation function
evaluate_model(model, X_val, y_val)
