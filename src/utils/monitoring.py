from prometheus_client import start_http_server, Summary, Gauge

TRAINING_TIME = Summary('training_seconds', 'Time spent training')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
MODEL_VERSION = Gauge('model_version', 'Model version', ['version'])

def start_monitoring(port=9100):
    start_http_server(port)