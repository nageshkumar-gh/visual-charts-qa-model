import os
import json
from PIL import Image

# Paths
#annotation_dir = 'Data/train/annotations'
#image_dir = 'Data/train/png'
#output_label_dir = 'Data/train/labels'

#annotation_dir = 'Data/test/annotations'
#image_dir = 'Data/test/png'
#output_label_dir = 'Data/test/labels'

annotation_dir = 'Data/val/annotations'
image_dir = 'Data/val/png'
output_label_dir = 'Data/val/labels'

os.makedirs(output_label_dir, exist_ok=True)

# Classes to extract
CLASS_MAP = {
    'title': 0,
    'x_axis_label': 1,
    'legend_label': 2
    # Add more if needed
}

# Convert to YOLO format
def convert_to_yolo(bbox, img_w, img_h):
    x_center = (bbox['x'] + bbox['w'] / 2) / img_w
    y_center = (bbox['y'] + bbox['h'] / 2) / img_h
    width = bbox['w'] / img_w
    height = bbox['h'] / img_h
    return x_center, y_center, width, height

# Process all annotations
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

    # Title
    title_info = ann.get('general_figure_info', {}).get('title', {})
    if 'bbox' in title_info:
        x, y, w, h = convert_to_yolo(title_info['bbox'], img_w, img_h)
        labels.append(f"{CLASS_MAP['title']} {x} {y} {w} {h}")

    # X-axis labels
    x_labels = ann.get('general_figure_info', {}).get('x_axis', {}).get('major_labels', {}).get('bboxes', [])
    for bbox in x_labels:
        x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
        labels.append(f"{CLASS_MAP['x_axis_label']} {x} {y} {w} {h}")

    # Legend items
    legend_items = ann.get('general_figure_info', {}).get('legend', {}).get('items', [])
    for item in legend_items:
        bbox = item.get('label', {}).get('bbox', {})
        x, y, w, h = convert_to_yolo(bbox, img_w, img_h)
        labels.append(f"{CLASS_MAP['legend_label']} {x} {y} {w} {h}")

    # Save label file
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

print("Annotations converted to YOLO format.")
