import json
import pandas as pd
import os

# JSON files to merge
json_files = [
    'Data/train/train_augmented.json',
    'Data/train/train_human.json'
]

# Location of table files
table_dir = 'Data/train/tables'

# Output file
output_dir = 'Data/train/input'
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, 'input_train_t5.csv')

# Helper: Flatten table CSV
def flatten_table(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return ' '.join(df.astype(str).apply(
            lambda row: ', '.join(f"{col}: {val}" for col, val in row.items()), axis=1))
    except Exception as e:
        print(f"Warning: Could not load table: {csv_path} ({e})")
        return ''

# Collect rows from all JSON files
all_rows = []

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_name = item['imgname']
        base_name = os.path.splitext(image_name)[0]
        table_path = os.path.join(table_dir, base_name + '.csv')
        table_text = flatten_table(table_path)

        input_text = f"question: {item['query']}"
        if table_text:
            input_text += f" table: {table_text}"

        all_rows.append({
            'input': input_text,
            'output': item['label'],
            'imgname': image_name
        })

# Create final DataFrame and save
df = pd.DataFrame(all_rows)
df.to_csv(output_csv_path, index=False)
print(f"{output_csv_path} created with {len(df)} rows.")

