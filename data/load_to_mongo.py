import os
from pathlib import Path
import io
from PIL import Image
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import random
import socket
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
IMAGES_PER_CLASS = 30  # 60 per split (30 normal, 30 glaucoma)
IMAGE_SIZE = (128, 128)
SPLITS = ['train', 'val', 'test']
MAX_RETRIES = 3
RETRY_DELAY = 2
THREADS = 5  # Number of parallel uploads

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

def process_and_upload_image(img_path, diagnosis, split, fs, db):
    """Processes and uploads a single image"""
    try:
        with open(img_path, "rb") as f:
            img = Image.open(io.BytesIO(f.read()))
            img = img.resize(IMAGE_SIZE).convert("RGB")
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=90)
            
            file_id = fs.put(
                img_byte_arr.getvalue(),
                filename=img_path.name,
                metadata={
                    "original_path": str(img_path),
                    "diagnosis": "glaucoma" if diagnosis == "1" else "normal",
                    "split": split,
                    "processing": f"resized_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
                }
            )
            
            db.images.insert_one({
                "file_id": file_id,
                "filename": img_path.name,
                "diagnosis": "glaucoma" if diagnosis == "1" else "normal",
                "split": split
            })
    except Exception as e:
        print(f"⚠️ Failed {img_path.name}: {str(e)}")

def upload_images(db, fs, dataset_root: Path):
    """Uploads images using threading for efficiency"""
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []
        
        for split in SPLITS:
            print(f"\n=== Processing {split} set ===")
            
            for diagnosis in ["0", "1"]:
                dir_path = dataset_root / split / diagnosis
                if not dir_path.exists():
                    print(f"⚠️ Missing directory: {dir_path}")
                    continue
                    
                images = list(dir_path.glob("*.*"))
                if len(images) < IMAGES_PER_CLASS:
                    print(f"⚠️ Not enough images in {dir_path} (found {len(images)}, need {IMAGES_PER_CLASS})")
                    continue
                    
                selected = random.sample(images, IMAGES_PER_CLASS)
                
                for img_path in selected:
                    futures.append(executor.submit(process_and_upload_image, img_path, diagnosis, split, fs, db))
                    
        for future in futures:
            future.result()

if __name__ == "__main__":
    try:
        client = get_mongo_connection()
        db = client[os.getenv("DB_NAME", "glaucoma_db")]
        fs = GridFS(db)
        
        data_root = Path(__file__).parent / "raw"
        upload_images(db, fs, data_root)
        
        # Verification
        print("\n=== Verification ===")
        for split in SPLITS:
            total = db.images.count_documents({"split": split})
            glaucoma = db.images.count_documents({"split": split, "diagnosis": "glaucoma"})
            normal = db.images.count_documents({"split": split, "diagnosis": "normal"})
            print(f"{split.upper()}: {total} total ({glaucoma} glaucoma, {normal} normal)")
        
        print("\n✅ Dataset upload complete!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
