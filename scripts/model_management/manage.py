import os
import shutil
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')

def archive_model(version):
    """Archive current model version"""
    src = os.path.join(MODEL_DIR, 'current/glaucoma_model_latest.h5')
    dest_dir = os.path.join(MODEL_DIR, f'archives/{version}')
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dest_dir, f'glaucoma_model_{version}.h5'))

def list_models():
    """List all available model versions"""
    print("Available models:")
    for root, dirs, files in os.walk(os.path.join(MODEL_DIR, 'archives')):
        for file in files:
            if file.endswith('.h5'):
                print(f"- {os.path.join(root, file)}")
    current = os.path.realpath(os.path.join(MODEL_DIR, 'current/glaucoma_model_latest.h5'))
    print(f"\nCurrent model: {current}")

def validate_model(path):
    """Validate model compatibility"""
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(path)
        print(f"Valid model: {path}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"Invalid model: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Management CLI")
    subparsers = parser.add_subparsers(dest='command')
    
    # Archive command
    archive_parser = subparsers.add_parser('archive')
    archive_parser.add_argument('version', help='Version number (e.g., v1.0)')
    
    # List command
    subparsers.add_parser('list')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('path', help='Path to model file')
    
    args = parser.parse_args()
    
    if args.command == 'archive':
        archive_model(args.version)
    elif args.command == 'list':
        list_models()
    elif args.command == 'validate':
        validate_model(args.path)
    else:
        parser.print_help()