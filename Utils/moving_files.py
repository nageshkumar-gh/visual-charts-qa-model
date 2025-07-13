import os
import shutil
import pandas as pd

# Base directories
base_dir = 'Data'
source_png_folder = os.path.join(base_dir, 'Dataset', 'png')
source_csv_folder = os.path.join(base_dir, 'Dataset', 'tables')

# Destination folders
destinations = {
    'train': {
        'png': os.path.join(base_dir, 'train', 'png'),
        'csv': os.path.join(base_dir, 'train', 'tables')
    },
    'val': {
        'png': os.path.join(base_dir, 'val', 'png'),
        'csv': os.path.join(base_dir, 'val', 'tables')
    },
    'test': {
        'png': os.path.join(base_dir, 'test', 'png'),
        'csv': os.path.join(base_dir, 'test', 'tables')
    }
}

# CSV file paths (split definitions)
csv_paths = {
    'train': 'Utils/output/train_split_files.csv',
    'val': 'Utils/output/val_split_files.csv',
    'test': 'Utils/output/test_split_files.csv'
}

# Copy PNG and CSV files based on filenames without extensions
for split, csv_path in csv_paths.items():
    df = pd.read_csv(csv_path)
    dest_png = destinations[split]['png']
    dest_csv = destinations[split]['csv']

    for base_name in df['filename']:
        # Copy PNG
        png_filename = base_name + '.png'
        src_png_path = os.path.join(source_png_folder, png_filename)
        dst_png_path = os.path.join(dest_png, png_filename)

        if os.path.exists(src_png_path):
            shutil.copy(src_png_path, dst_png_path)
        else:
            print(f"PNG file not found: {src_png_path}")

        # Copy CSV
        csv_filename = base_name + '.csv'
        src_csv_path = os.path.join(source_csv_folder, csv_filename)
        dst_csv_path = os.path.join(dest_csv, csv_filename)

        if os.path.exists(src_csv_path):
            shutil.copy(src_csv_path, dst_csv_path)
        else:
            print(f"Table file not found: {src_csv_path}")


folder_train_path = 'Data/train/png'
folder_val_path = 'Data/val/png'
folder_test_path = 'Data/test/png'

file_count = len([f for f in os.listdir(folder_train_path) if os.path.isfile(os.path.join(folder_train_path, f))])
print(f"Total files in '{folder_train_path}': {file_count}")

file_count = len([f for f in os.listdir(folder_val_path) if os.path.isfile(os.path.join(folder_val_path, f))])
print(f"Total files in '{folder_val_path}': {file_count}")

file_count = len([f for f in os.listdir(folder_test_path) if os.path.isfile(os.path.join(folder_test_path, f))])
print(f"Total files in '{folder_test_path}': {file_count}")
