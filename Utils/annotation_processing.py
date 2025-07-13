import os
import json
import csv

# Set the path to the folder containing your JSON files
json_folder = 'Data/Dataset/annotations'
output_csv = 'Utils/output/output.csv'

data = []

# Loop through all JSON files in the folder
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        filepath = os.path.join(json_folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                chart_type = content.get('type', 'N/A')  # default to 'N/A' if 'type' not found
                
                # Remove extension (.json or any)
                name_without_ext = os.path.splitext(filename)[0]

                data.append({'filename': name_without_ext, 'type': chart_type})
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Write to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'type'])
    writer.writeheader()
    writer.writerows(data)

print(f"CSV created successfully: {output_csv}")

