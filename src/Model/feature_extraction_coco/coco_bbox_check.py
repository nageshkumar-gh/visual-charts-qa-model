import os
import json

# Path to your annotation folder
annotation_folder = "Data/dataset/annotations"

# Loop through all JSON files in the folder
for filename in os.listdir(annotation_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(annotation_folder, filename)

        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                chart_type = data.get("type", "")
                gf = data.get("general_figure_info", {})
                #gf = data.get("chart", {})  # Or change key if the structure is different

                # Get y-axis label bounding boxes
                #for tick in gf.get("y_axis", {}).get("label", {}).get("bbox", []):
                 #   print(f"{filename} - YAxistitle---", tick)

                # Get x-axis label bounding boxes
                for tick in gf.get("x_axis", {}):#.get("label", {}):#.get("bbox", []):
                    print(f"{filename} - xAxistitle---", tick)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
