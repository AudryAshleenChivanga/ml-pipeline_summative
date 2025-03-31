import hashlib

def verify_model_integrity(model_path):
    """Verify model file integrity using SHA-256"""
    known_hashes = {
        'CNN_Model_1.h5': 'your_model_hash_here'
    }
    
    filename = os.path.basename(model_path)
    if filename not in known_hashes:
        print(f"Warning: No known hash for {filename}")
        return False
    
    with open(model_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash == known_hashes[filename]:
        print(f"Model integrity verified: {filename}")
        return True
    else:
        print(f"Model integrity check failed for {filename}")
        return False