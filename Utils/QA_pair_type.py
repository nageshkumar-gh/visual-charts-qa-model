import json
import os

# Base directory
base_dir = 'Data'

# Dataset configurations: (subdir, filename)
datasets = [
    ('train', 'train_augmented.json'),
    ('train', 'train_human.json'),
    ('val', 'val_augmented.json'),
    ('val', 'val_human.json'),
    ('test', 'test_augmented.json'),
    ('test', 'test_human.json'),
]

def categorize_label(label):
    label = label.strip()
    if label.lower() == "yes":
        return "yes"
    elif label.lower() == "no":
        return "no"
    elif label.isdigit():
        return "integer"
    else:
        return "string"

# Process each dataset
for subdir, filename in datasets:
    path = os.path.join(base_dir, subdir, filename)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {"yes": 0, "no": 0, "integer": 0, "string": 0}

    for item in data:
        label = item.get("label", "").strip()
        category = categorize_label(label)
        stats[category] += 1

    print(f"\nDataset: {filename}")
    print(f"  Yes:     {stats['yes']}")
    print(f"  No:      {stats['no']}")
    print(f"  Integer: {stats['integer']}")
    print(f"  String:  {stats['string']}")
