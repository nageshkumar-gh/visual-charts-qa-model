import json

def check_json_validity(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded: {json_path}")
        print(f"Top-level keys: {list(data.keys())}")
        return data
    except Exception as e:
        print(f" Error reading {json_path}: {e}")

data = check_json_validity("Data/train/train_coco_annotations.json")

