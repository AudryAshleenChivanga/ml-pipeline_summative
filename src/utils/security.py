import os

def validate_token(token):
    return token == os.getenv("API_TOKEN")

def rbac_check(user, action):
    roles = {
        "retrain": ["admin", "data-scientist"],
        "predict": ["user", "admin"]
    }
    return user.role in roles.get(action, [])