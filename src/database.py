#!/usr/bin/env python3
from pathlib import Path
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS

# Load .env from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def get_mongo_client():
    """Secure MongoDB connection with explicit database selection"""
    mongodb_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME", "glaucoma_db")  # Default fallback
    
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in .env")
    
    try:
        client = MongoClient(
            mongodb_uri,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000
        )
        
        db = client.get_database(db_name)
        client.admin.command('ping')  
        return client, GridFS(db)
    except Exception as e:
        raise ConnectionError(f"MongoDB connection failed: {str(e)}")

if __name__ == "__main__":
    print("Testing MongoDB connection...")
    try:
        client, fs = get_mongo_client()
        print("✅ Connection successful!")
        print(f"Database: {client.get_database().name}")
        print(f"Collections: {client.get_database().list_collection_names()}")
        client.close()
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")