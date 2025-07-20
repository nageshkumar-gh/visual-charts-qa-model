import json

# JSON file paths
json_human = 'Data/Dataset/human.json'
json_aug = 'Data/Dataset/augmented.json'
train_json_human = 'Data/train/train_human.json'
train_json_aug = 'Data/train/train_augmented.json'
test_json_human = 'Data/test/test_human.json'
test_json_aug = 'Data/test/test_augmented.json'
val_json_human = 'Data/val/val_human.json'
val_json_aug = 'Data/val/val_augmented.json'

# List of all JSONs to check
json_files = [
    json_human,
    json_aug,
    train_json_human,
    train_json_aug,
    test_json_human,
    test_json_aug,
    val_json_human,
    val_json_aug
]

# Load and count entries
for path in json_files:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"{path} has {len(data)} entries.")
    except FileNotFoundError:
        print(f"{path} not found.")
    except json.JSONDecodeError:
        print(f"{path} is not a valid JSON file.")
