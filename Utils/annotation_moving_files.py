import os
import shutil
import pandas as pd

# Base directories
base_dir = 'Data'
source_anno_dir = os.path.join(base_dir, 'Dataset', 'annotations')

# Target split config
splits = {
    'train': {
        'csv': 'Utils/output/train_split_files.csv',
        'anno_dir': os.path.join(base_dir, 'train', 'annotations')
    },
    'val': {
        'csv': 'Utils/output/val_split_files.csv',
        'anno_dir': os.path.join(base_dir, 'val', 'annotations')
    },
    'test': {
        'csv': 'Utils/output/test_split_files.csv',
        'anno_dir': os.path.join(base_dir, 'test', 'annotations')
    }
}

# Print initial count of files in source
initial_source_count = len([
    f for f in os.listdir(source_anno_dir) if f.endswith('.json')
])
print(f"Initial files in source folder ({source_anno_dir}): {initial_source_count}")

# Ensure target folders exist
for split in splits:
    os.makedirs(splits[split]['anno_dir'], exist_ok=True)

# Copy annotation files from source to each target split
for split, info in splits.items():
    df = pd.read_csv(info['csv'])
    copy_count = 0
    for name in df['filename']:
        base_name = os.path.splitext(name)[0]  # remove .png if present
        json_name = base_name + '.json'

        src_path = os.path.join(source_anno_dir, json_name)
        dst_path = os.path.join(info['anno_dir'], json_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copy_count += 1
        else:
            print(f"Warning: {src_path} not found.")

    print(f"{split.capitalize()} set: Copied {copy_count} files.")

# Print final file count in each target folder
print("\nFiles in each target folder after copy:")
for split, info in splits.items():
    count = len([
        f for f in os.listdir(info['anno_dir']) if f.endswith('.json')
    ])
    print(f"{split.capitalize()} folder ({info['anno_dir']}): {count} files")

# Show source folder count again (should be unchanged)
final_source_count = len([
    f for f in os.listdir(source_anno_dir) if f.endswith('.json')
])
print(f"\nFinal files in source folder: {final_source_count}")
