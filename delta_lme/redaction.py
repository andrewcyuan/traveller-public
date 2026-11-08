import hashlib
import base64

def strip_images(value):
    if isinstance(value, dict):
        return {k: strip_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_images(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"data_url_sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value
