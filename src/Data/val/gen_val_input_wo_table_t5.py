import json
import pandas as pd
import os

# JSON files to merge
json_files = [
    'Data/val/val_augmented.json',
    'Data/val/val_human.json'
]

# Folder locations
annotation_dir = 'Data/val/annotations'
output_dir = 'Data/val/input'
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(output_dir, 'input_val_wo_table_t5.csv')

# Helper: Load chart type and title from annotation JSON
def load_annotation_info(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chart_type = data.get('type', '')
            title = data.get('general_figure_info', {}).get('title', {}).get('text', '')
            axis_y = data.get('general_figure_info', {}).get('y_axis', {}).get('label', {}).get('text', '')
            axis_x = data.get('general_figure_info', {}).get('x_axis', {}).get('major_labels', {}).get('values', '')
            legend_items = data.get('general_figure_info', {}).get('legend', {}).get('items', [])
            legends = [item['label']['text'] for item in legend_items if 'label' in item and 'text' in item['label']]
            return chart_type, title, axis_y, axis_x, legends
    except Exception as e:
        print(f"Warning: Could not read annotation {json_path} ({e})")
        return 'N/A', 'N/A', 'N/A', 'N/A'

# Collect all rows
all_rows = []

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        image_name = item['imgname']
        base_name = os.path.splitext(image_name)[0]

        # Path to annotation
        annotation_path = os.path.join(annotation_dir, base_name + '.json')

        # Load components
        chart_type, title, axis_y, axis_x, legends = load_annotation_info(annotation_path)

        # Build input field without table
        input_text = f"question: {item['query']} chart_type: {chart_type} title: {title} x_axis_title: {axis_x} y_axis_title: {axis_y} legends: {legends}"

        all_rows.append({
            'input': input_text,
            'output': item['label'],
            'imgname': image_name
        })

# Save to CSV
df = pd.DataFrame(all_rows)
df.to_csv(output_csv_path, index=False)
print(f"{output_csv_path} created with {len(df)} rows.")
