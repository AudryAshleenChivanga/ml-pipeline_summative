import os

class ModelRetrainer:
    def __init__(self):
        self.model = None
        self.model_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../models'
        ))
    
    def load_model(self):
        model_path = os.path.join(self.model_dir, 'current/glaucoma_model_latest.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"Loaded model: {model_path}")