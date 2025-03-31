import os
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv
from pymongo import MongoClient
from gridfs import GridFS
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

class MongoDataLoader:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client[os.getenv("DB_NAME")]
        self.fs = GridFS(self.db)

    def load_dataset(self, split='train'):
        """Load images and labels from MongoDB"""
        X, y = [], []
        for record in self.db.images.find({'split': split}):
            img_data = self.fs.get(record['file_id']).read()
            img = Image.open(io.BytesIO(img_data))
            X.append(np.array(img))
            y.append(1 if record['diagnosis'] == 'glaucoma' else 0)
        return np.array(X), np.array(y)

    def get_class_distribution(self):
        """Return counts for each class"""
        return {
            'train': {
                'glaucoma': self.db.images.count_documents({'split': 'train', 'diagnosis': 'glaucoma'}),
                'normal': self.db.images.count_documents({'split': 'train', 'diagnosis': 'normal'})
            },
            'test': {
                'glaucoma': self.db.images.count_documents({'split': 'test', 'diagnosis': 'glaucoma'}),
                'normal': self.db.images.count_documents({'split': 'test', 'diagnosis': 'normal'})
            }
        }