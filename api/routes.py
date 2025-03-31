from flask import Blueprint, jsonify, request
from src.data.mongo_connector import MongoRetrainer
from src.models.retrain_service import ModelRetrainer
from src.utils.security import validate_token
from src.utils.monitoring import TRAINING_TIME, MODEL_ACCURACY

retrain_bp = Blueprint('retrain', __name__)

@retrain_bp.route('/retrain', methods=['POST'])
@TRAINING_TIME.time()
def handle_retraining():
    # Authentication
    if not validate_token(request.headers.get('Authorization')):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Initialize components
    mongo = MongoRetrainer()
    retrainer = ModelRetrainer()
    
    try:
        # Load current model
        retrainer.load_model("models/glaucoma_model_latest.h5")
        
        # Get new data
        new_data = list(mongo.get_latest_data())
        if not new_data:
            return jsonify({"status": "No new data available"}), 200
        
        # Prepare and train
        train_ds, val_ds = retrainer.prepare_datasets(new_data)
        history, version = retrainer.retrain(train_ds, val_ds)
        MODEL_ACCURACY.set(history.history['val_accuracy'][-1])
        
        # Update processed status
        mongo.mark_as_processed([doc["_id"] for doc in new_data])
        
        return jsonify({
            "status": "success",
            "model_version": version,
            "metrics": {
                "accuracy": history.history['accuracy'],
                "val_accuracy": history.history['val_accuracy'],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500