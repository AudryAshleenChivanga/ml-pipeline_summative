import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def preprocess_images(image_data):
    """
    Preprocesses the input image data by normalizing pixel values.

    Args:
        image_data (list or np.array): The image data to be processed. 
            The input is expected to be a list or array of image data (e.g., loaded as RGB images).

    Returns:
        np.array: The processed image data with pixel values normalized to the range [0, 1].
    """
    # Normalizing the image data to the range [0, 1]
    processed_images = np.array(image_data) / 255.0
    return processed_images

def create_data_augmentation_pipeline(X_train):
    """
    Creates a data augmentation pipeline using Keras' ImageDataGenerator.

    Args:
        X_train (np.array): The training image data. It should be a NumPy array 
            with shape (num_images, height, width, channels).

    Returns:
        ImageDataGenerator: A data augmentation generator that applies random transformations to the images.
    """
    # Defining the data augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=20,             
        width_shift_range=0.2,         
        height_shift_range=0.2,       
        shear_range=0.2,               
        zoom_range=0.2,               
        horizontal_flip=True,          
        fill_mode='nearest'            
    )
    
    # Fit the generator to the training data
    datagen.fit(X_train)
    
    return datagen

def one_hot_encode_labels(labels, num_classes=2):
    """
    One-hot encodes the labels for categorical classification.

    Args:
        labels (list or np.array): The labels to be one-hot encoded.
        num_classes (int): The number of classes (default is 2 for binary classification).

    Returns:
        np.array: The one-hot encoded labels.
    """
    return to_categorical(labels, num_classes)
