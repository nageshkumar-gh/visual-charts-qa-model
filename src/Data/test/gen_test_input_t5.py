import json
import pandas as pd
import os

# JSON files to merge
json_files = [
    'Data/test/test_augmented.json',
    'Data/test/test_human.json'
]

# Folder locations
table_dir = 'Data/test/tables'
annotation_dir = 'Data/test/annotations'
output_dir = 'Data/test/input'
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, 'input_test_annot_t5.csv')

# Helper: Flatten table CSV
def flatten_table(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return ' '.join(df.astype(str).apply(
            #lambda row: '; '.join(f"{col}: {val}," for col, val in row.items()), axis=1))
            lambda row: ', '.join(f"{col}: {val}" for col, val in row.items()) + ';',
        axis=1))
    except Exception as e:
        print(f"Warning: Could not load table: {csv_path} ({e})")
        return ''

# Helper: Load chart type and title from annotation JSON
def load_annotation_info(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chart_type = data.get('type', '')
            title = data.get('general_figure_info', {}).get('title', {}).get('text', '')
            axis_y = data.get('general_figure_info', {}).get('y_axis', {}).get('label', {}).get('text', '')
            axis_x = data.get('general_figure_info', {}).get('x_axis', {}).get('major_labels', {}).get('values', '')
            return chart_type, title, axis_y, axis_x
    except Exception as e:
        print(f"Warning: Could not read annotation {json_path} ({e})")
        return 'N/A', 'N/A'

# Collect all rows
all_rows = []

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_name = item['imgname']
        base_name = os.path.splitext(image_name)[0]

        # Paths to table and annotation
        table_path = os.path.join(table_dir, base_name + '.csv')
        annotation_path = os.path.join(annotation_dir, base_name + '.json')

        # Load components
        table_text = flatten_table(table_path)
        chart_type, title, axis_y, axis_x = load_annotation_info(annotation_path)

        # Build input field
        input_text = f"question: {item['query']}"
        if table_text:
            input_text += f" table: {table_text}"
        input_text += f" chart_type: {chart_type} title: {title} x_axis_title: {axis_x} y_axis_title: {axis_y}"

        all_rows.append({
            'input': input_text,
            'output': item['label'],
            'imgname': image_name
        })

# Save to CSV
df = pd.DataFrame(all_rows)
df.to_csv(output_csv_path, index=False)
print(f"{output_csv_path} created with {len(df)} rows.")
