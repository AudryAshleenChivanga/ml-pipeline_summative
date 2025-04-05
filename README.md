# ML Pipelines and Model Retraining for Pre-trained Glaucoma Prediction Model

## Task Overview
The **ML Pipelines and Model Retraining for Pre-trained Glaucoma Prediction Model** project focuses on enhancing a pre-trained convolutional neural network (CNN) to improve glaucoma detection accuracy. This was achieved by applying data augmentation techniques, optimizing the architecture with regularization methods, and retraining the model on new data. The final model was evaluated using performance metrics such as accuracy, precision, recall, and F1-score.

## Project front-end is to be found here : 
https://github.com/AudryAshleenChivanga/pipeline_summative-frontend

## Video recording link : 
https://vimeo.com/1072370971?share=copy
## Live Project Deployed : 
https://audryashleenchivanga.github.io/pipeline_summative-frontend/

## Dataset
The project utilizes a dataset containing retinal fundus images labeled for glaucoma detection. The dataset includes:

- **Image ID:** A unique identifier for each image.
- **Fundus Image:** The actual retinal image.
- **Glaucoma Label:** A binary label indicating glaucoma presence (1) or absence (0).
- **Patient Age:** The age of the patient.
- **Other Metadata:** Additional clinical features that may assist in prediction.

## Project Structure
ML-PIPELINE_SUMMATIVE/
````
├── data/
│   ├──mongo/migrations
│   └── load_to_mongo.py
|
├── notebook/
│   └── ML_Pipeline_Retraining_Notebook.ipynb
|
├── src/
│   └── preprocessing.py
│   └── model.py
│   └── prediction.py
|   └── database.py
|   └── data_loader.py
|   └── .ipynb_checkpoints
|
├── .gitignore
|
├── main.py
|
├── requirements.in
├── requirements.txt
|
└── models/
    ├── CNN_Model_Updated.h5
    └── _model_base.h5

````
## Installation

Clone the repository:
```sh
git clone https://github.com/AudryAshleenChivanga/ml-pipeline_summative/tree/main
cd ml-pipeline_summative
````
Create a virtual environment and activate it:
````
python3 -m venv venv
source venv/bin/activate
````

Install the required dependencies:
````
pip install -r requirements.txt
````
# Data Loading and Preprocessing
Loading Data into MongoDB
The dataset was stored in MongoDB to enable efficient querying and retrieval of image data. The following steps were taken:

Convert Images to Base64: Images were converted into Base64 strings for easy storage.

Insert into MongoDB: A Python script using pymongo was used to store images and their corresponding labels in a MongoDB collection.

Query for Training: Images were retrieved using MongoDB queries and decoded for use in the model.

Image Preprocessing
To ensure optimal model performance, several preprocessing steps were applied:

Resizing: Images were resized to 224x224 pixels to match the input requirements of the pre-trained CNN.

Normalization: Pixel values were normalized to the range [0,1] for stable training.

Data Augmentation: Techniques such as rotation, flipping, and contrast adjustments were applied to improve model generalization.

Splitting: The dataset was split into training (80%) and testing (20%) sets.

Regularization: Dropout layers were added to prevent overfitting.

Optimization: The Adam optimizer was used with a learning rate of 0.0001.

Loss Function: Binary cross-entropy loss was used for classification.

To train the model, run:
''''
python src/model_training.py
''''
# Model Evaluation
## The trained model was evaluated using the following metrics:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Check initial results in reports/

Predictions
To make predictions on new images, use:
````
python src/prediction.py --image_path /path/to/image.jpg
````
# Technologies Used
## This project utilizes:

Machine Learning: TensorFlow, Keras

Data Storage: MongoDB

Data Processing: NumPy, OpenCV, Pandas

Model Training & Evaluation: Scikit-learn, Matplotlib

Backend : FastAPI

# Author :
**Audry Ashleen Chivanga**