import os
import json

# Set image size (replace with actual if known)
img_width = 800
img_height = 600

# Class mapping for YOLO class IDs
class_map = {
    "bar": 0,
    "x_tick": 1,
    "y_tick": 2,
    "y_label": 3,
    "legend": 4,
    "title": 5
}

def normalize_bbox(x, y, w, h):
    x_center = min(max((x + w / 2) / img_width, 0.0), 1.0)
    y_center = min(max((y + h / 2) / img_height, 0.0), 1.0)
    w_norm = min(max(w / img_width, 0.0), 1.0)
    h_norm = min(max(h / img_height, 0.0), 1.0)
    return x_center, y_center, w_norm, h_norm

def convert_annotation(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    yolo_lines = []

    # --- Y-axis ticks ---
    for bbox in data.get("general_figure_info", {}).get("y_axis", {}).get("major_labels", {}).get("bboxes", []):
        x_c, y_c, w_n, h_n = normalize_bbox(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        yolo_lines.append(f"{class_map['y_tick']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- Y-axis label (handle dict or list) ---
    y_label_box = data.get("general_figure_info", {}).get("y_axis", {}).get("label", {}).get("bbox", None)
    if isinstance(y_label_box, dict):
        x_c, y_c, w_n, h_n = normalize_bbox(y_label_box["x"], y_label_box["y"], y_label_box["w"], y_label_box["h"])
        yolo_lines.append(f"{class_map['y_label']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
    elif isinstance(y_label_box, list):
        for bbox in y_label_box:
            x_c, y_c, w_n, h_n = normalize_bbox(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
            yolo_lines.append(f"{class_map['y_label']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- X-axis ticks ---
    for bbox in data.get("general_figure_info", {}).get("x_axis", {}).get("major_labels", {}).get("bboxes", []):
        x_c, y_c, w_n, h_n = normalize_bbox(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        yolo_lines.append(f"{class_map['x_tick']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- Bars (models) ---
    for model in data.get("models", []):
        for bbox in model.get("bboxes", []):
            x_c, y_c, w_n, h_n = normalize_bbox(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
            yolo_lines.append(f"{class_map['bar']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- Title (handle dict or list) ---
    title_bbox = data.get("general_figure_info", {}).get("title", {}).get("bbox", None)
    if isinstance(title_bbox, dict):
        x_c, y_c, w_n, h_n = normalize_bbox(title_bbox["x"], title_bbox["y"], title_bbox["w"], title_bbox["h"])
        yolo_lines.append(f"{class_map['title']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
    elif isinstance(title_bbox, list):
        for bbox in title_bbox:
            x_c, y_c, w_n, h_n = normalize_bbox(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
            yolo_lines.append(f"{class_map['title']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # --- Legend (handle dict or list) ---
    legend_bbox = data.get("general_figure_info", {}).get("legend", {}).get("bbox", None)
    if isinstance(legend_bbox, dict):
        x_c, y_c, w_n, h_n = normalize_bbox(legend_bbox["x"], legend_bbox["y"], legend_bbox["w"], legend_bbox["h"])
        yolo_lines.append(f"{class_map['legend']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
    elif isinstance(legend_bbox, list):
        for bbox in legend_bbox:
            x_c, y_c, w_n, h_n = normalize_bbox(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
            yolo_lines.append(f"{class_map['legend']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

    # Save label file
    with open(output_path, 'w') as f_out:
        for line in yolo_lines:
            f_out.write(line + '\n')

# --- Process all splits ---
splits = ["train", "val", "test"]

for split in splits:
    annotation_dir = f'Data/{split}/annotations'
    output_label_dir = f'Data/{split}/labels'
    os.makedirs(output_label_dir, exist_ok=True)

    print(f"\n Processing {split} set...")

    for filename in os.listdir(annotation_dir):
        if filename.endswith('.json'):
            base_name = os.path.splitext(filename)[0]
            json_path = os.path.join(annotation_dir, filename)
            label_output_path = os.path.join(output_label_dir, base_name + '.txt')
            try:
                convert_annotation(json_path, label_output_path)
                print(f"{filename} → {base_name}.txt")
            except Exception as e:
                print(f" Failed: {filename} — {e}")

print("\n All splits processed.")
