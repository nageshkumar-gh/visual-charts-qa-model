import json
import pandas as pd
import os

# JSON files to merge
json_files = [
    'Data/val/val_augmented.json',
    'Data/val/val_human.json'
]

# Folder locations
table_dir = 'Data/val/tables'
output_dir = 'Data/val/input'
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, 'input_val_wo_annot_t5.csv')

# Helper: Flatten table CSV
def flatten_table(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return ' '.join(df.astype(str).apply(
            lambda row: ', '.join(f"{col}: {val}" for col, val in row.items()) + ';',
        axis=1))
    except Exception as e:
        print(f"Warning: Could not load table: {csv_path} ({e})")
        return ''

# Collect all rows
all_rows = []

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_name = item['imgname']
        base_name = os.path.splitext(image_name)[0]

        # Path to table
        table_path = os.path.join(table_dir, base_name + '.csv')

        # Load table
        table_text = flatten_table(table_path)

        # Build input field
        input_text = f"question: {item['query']}"
        if table_text:
            input_text += f" table: {table_text}"

        all_rows.append({
            'input': input_text,
            'output': item['label'],
            'imgname': image_name
        })

# Save to CSV
df = pd.DataFrame(all_rows)
df.to_csv(output_csv_path, index=False)
print(f"{output_csv_path} created with {len(df)} rows.")