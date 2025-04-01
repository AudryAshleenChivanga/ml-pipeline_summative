import os
from pymongo import MongoClient
from dotenv import load_dotenv

# environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Collections
images_collection = db["predictions"]
retrain_collection = db["retrain_logs"]