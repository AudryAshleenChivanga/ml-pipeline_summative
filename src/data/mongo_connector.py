from pymongo import MongoClient
import os
from datetime import datetime

class MongoRetrainer:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URI"))
        self.db = self.client[os.getenv("MONGO_DB", "glaucoma_db")]
        
    def get_latest_data(self):
        """Fetch unprocessed data"""
        return self.db.fundus_images.find({
            "processed": False,
            "quality_check": {"$exists": True}
        })
    
    def mark_as_processed(self, object_ids):
        """Update processed status"""
        self.db.fundus_images.update_many(
            {"_id": {"$in": object_ids}},
            {"$set": {
                "processed": True,
                "processed_at": datetime.utcnow()
            }}
        )