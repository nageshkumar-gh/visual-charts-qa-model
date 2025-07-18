import os
import json
from PIL import Image

# === Paths ===
annotation_dir = 'Data/val/annotations'  # Change to train/test/val
image_dir = 'Data/val/png'
output_label_dir = 'Data/val/labels'

os.makedirs(output_label_dir, exist_ok=True)

# === YOLO Class Mapping ===
CLASS_MAP = {
    "title": 0,
    "x_axis": 1,
    "y_axis": 2,
    "legend": 3,
    "value": 4,
    "label": 5
}

# === Normalize Bounding Boxes ===
def convert_to_yolo(bbox, img_w, img_h):
    x_center = (bbox['x'] + bbox['w'] / 2) / img_w
    y_center = (bbox['y'] + bbox['h'] / 2) / img_h
    width = bbox['w'] / img_w
    height = bbox['h'] / img_h
    return x_center, y_center, width, height

# === Process All Annotations ===
for ann_file in os.listdir(annotation_dir):
    if not ann_file.endswith('.json'):
        continue

    base_name = os.path.splitext(ann_file)[0]
    img_path = os.path.join(image_dir, base_name + '.png')
    label_path = os.path.join(output_label_dir, base_name + '.txt')
    ann_path = os.path.join(annotation_dir, ann_file)

    if not os.path.exists(img_path):
        print(f"Skipping {ann_file}: image not found.")
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    with open(ann_path, 'r', encoding='utf-8') as f:
        ann = json.load(f)

    labels = []
    chart_type = ann.get("type")

    # --- Title ---
    title_info = ann.get('general_figure_info', {}).get('title', {})
    if 'bbox' in title_info:
        x, y, w, h = convert_to_yolo(title_info['bbox'], img_w, img_h)
        labels.append(f"{CLASS_MAP['title']} {x} {y} {w} {h}")

    # --- Legend Items ---
    legend_items = ann.get('general_figure_info', {}).get('legend', {}).get('items', [])
    for item in legend_items:
        bbox = item.get('label', {}).get('bbox', None)
        if bbox:
            x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
            labels.append(f"{CLASS_MAP['legend']} {x} {y} {w} {h}")

    # --- X-axis line ---
    x_axis_bbox = ann.get('general_figure_info', {}).get('general_figure_info', {}).get('x_axis', {}).get('major_labels',{}).get('bboxes', [])
    if x_axis_bbox:
        x, y, w, h = convert_to_yolo(x_axis_bbox, img_w, img_h)
        labels.append(f"{CLASS_MAP['x_axis']} {x} {y} {w} {h}")

    # --- Y-axis line ---
    y_axis_bboxes = ann.get('general_figure_info', {}).get('general_figure_info', {}).get('y_axis', {}).get('major_labels',{}).get('bboxes', [])
    if isinstance(y_axis_bboxes, list):
        for bbox in y_axis_bboxes:
            x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
            labels.append(f"{CLASS_MAP['y_axis']} {x} {y} {w} {h}")
    elif isinstance(y_axis_bboxes, dict):
        x, y, w, h = convert_to_yolo(y_axis_bboxes, img_w, img_h)
        labels.append(f"{CLASS_MAP['y_axis']} {x} {y} {w} {h}")

    # === PIE CHART ===
    if chart_type == "pie":
        for model in ann.get('models', []):
            text_bbox = model.get("text_bbox", None)
            if text_bbox:
                x, y, w, h = convert_to_yolo(text_bbox, img_w, img_h)
                labels.append(f"{CLASS_MAP['label']} {x} {y} {w} {h}")
                labels.append(f"{CLASS_MAP['value']} {x} {y} {w} {h}")

    # === LINE CHART ===
    elif chart_type == "line":
        x_labels = ann.get('general_figure_info', {}).get('x_axis', {}).get('major_labels', {}).get('bboxes', [])
        for bbox in x_labels:
            x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
            labels.append(f"{CLASS_MAP['label']} {x} {y} {w} {h}")
        for model in ann.get('models', []):
            for bbox in model.get('bboxes', []):
                x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
                labels.append(f"{CLASS_MAP['value']} {x} {y} {w} {h}")

    # === HORIZONTAL BAR CHART ===
    elif chart_type == "h_bar":
        x_labels = ann.get('general_figure_info', {}).get('x_axis', {}).get('major_labels', {}).get('bboxes', [])
        for bbox in x_labels:
            x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
            labels.append(f"{CLASS_MAP['label']} {x} {y} {w} {h}")
        for model in ann.get('models', []):
            for bbox in model.get('bboxes', []):
                x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
                labels.append(f"{CLASS_MAP['value']} {x} {y} {w} {h}")

    # === VERTICAL BAR CHART ===
    elif chart_type == "v_bar":
        x_labels = ann.get('general_figure_info', {}).get('x_axis', {}).get('major_labels', {}).get('bboxes', [])
        for bbox in x_labels:
            x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
            labels.append(f"{CLASS_MAP['label']} {x} {y} {w} {h}")
        y_labels = ann.get('general_figure_info', {}).get('y_axis', {}).get('major_labels', {}).get('bboxes', [])
        for bbox in y_labels:
            x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
            labels.append(f"{CLASS_MAP['label']} {x} {y} {w} {h}")
        for model in ann.get('models', []):
            for bbox in model.get('bboxes', []):
                x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
                labels.append(f"{CLASS_MAP['value']} {x} {y} {w} {h}")

    # === Save YOLO label file ===
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

print(" All annotations converted to YOLO format.")
