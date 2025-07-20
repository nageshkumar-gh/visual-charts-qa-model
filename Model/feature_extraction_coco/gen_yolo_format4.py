import os
import json
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
annotation_dir = "Data/val/annotations"
image_dir = "Data/val/png"
output_label_dir = "Data/val/yolo_labels"
labels="Data/val/labels"
output_class_file = os.path.join(labels, "classes.txt")
os.makedirs(output_label_dir, exist_ok=True)

# === CLASS MAPPING ===
objects_classes = {
    'ChartTitle': 0,
    'PlotArea': 1,
    'LegendLabel': 2,
    'xAxisLabel': 3,
    'yAxisLabel': 4,
    'PieLabel': 5,
    'v_bar': 6,
    'h_bar': 7,
    'line': 8,
    'yAxisTitle': 9
}
# Save classes.txt
with open(output_class_file, "w") as f:
    for label in sorted(objects_classes, key=objects_classes.get):
        f.write(f"{label}\n")

def convert_to_yolo_format(bbox, img_w, img_h):
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    if w <= 0 or h <= 0:
        return None
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

# === PROCESS ANNOTATIONS ===
for filename in tqdm(os.listdir(annotation_dir), desc="Converting annotations"):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(annotation_dir, filename)
    with open(path) as f:
        ann = json.load(f)

    base = filename.replace(".json", "")
    img_file = f"{base}.png"
    image_path = os.path.join(image_dir, img_file)

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except FileNotFoundError:
        print(f"Image not found: {image_path}, skipping.")
        continue

    label_lines = []
    chart_type = ann.get("type", "")
    gfi = ann.get("general_figure_info", {})

    # Add functionally all valid bbox fields
    def add_yolo_annotation(label, bbox):
        if label not in objects_classes:
            return
        converted = convert_to_yolo_format(bbox, width, height)
        if converted:
            class_id = objects_classes[label]
            label_lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in converted))

    # Chart Title
    if "title" in gfi and "bbox" in gfi["title"]:
        add_yolo_annotation("ChartTitle", gfi["title"]["bbox"])

    # Plot Area
    if "figure_info" in gfi and "bbox" in gfi["figure_info"]:
        add_yolo_annotation("PlotArea", gfi["figure_info"]["bbox"])

    # Legend Labels
    for item in gfi.get("legend", {}).get("items", []):
        if "label" in item and "bbox" in item["label"]:
            add_yolo_annotation("LegendLabel", item["label"]["bbox"])

    # X Axis Labels
    for tick in gfi.get("x_axis", {}).get("major_labels", {}).get("bboxes", []):
        add_yolo_annotation("xAxisLabel", tick)

    # Y Axis Labels
    for tick in gfi.get("y_axis", {}).get("major_labels", {}).get("bboxes", []):
        add_yolo_annotation("yAxisLabel", tick)

    # Y Axis Title
    for tick in gfi.get("y_axis", {}).get("label", {}).get("bbox", []):
        add_yolo_annotation("yAxisTitle", tick)

    # Models
    for model in ann.get("models", []):
        if chart_type == "pie":
            if "text_bbox" in model:
                add_yolo_annotation("PieLabel", model["text_bbox"])
        elif chart_type == "v_bar":
            for bbox in model.get("bboxes", []):
                add_yolo_annotation("v_bar", bbox)
        elif chart_type == "h_bar":
            for bbox in model.get("bboxes", []):
                add_yolo_annotation("h_bar", bbox)
        elif chart_type == "line":
            for bbox in model.get("bboxes", []):
                add_yolo_annotation("line", bbox)

    # Write YOLO label file
    label_file_path = os.path.join(output_label_dir, f"{base}.txt")
    with open(label_file_path, "w") as out_f:
        out_f.write("\n".join(label_lines))

print(f"\n YOLO label files saved to: {output_label_dir}")
