import os
from flask import Flask
from api.routes import retrain_bp

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create Flask application
app = Flask(__name__)

# Register blueprints
app.register_blueprint(retrain_bp, url_prefix='/api/v1')

if __name__ == '__main__':
    app.run(
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'false').lower() == 'true'
    )