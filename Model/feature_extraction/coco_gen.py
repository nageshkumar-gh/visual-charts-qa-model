import os
import json
from PIL import Image
from tqdm import tqdm

# Paths
annotation_dir = "Data/val/annotations"
image_dir = "Data/val/png"
output_path = "Data/val/val_coco_annotations.json"

# Class mapping
objects_classes = {
    'Legend': 0, 'yAxisTitle': 1, 'ChartTitle': 2, 'xAxisTitle': 3, 'LegendPreview': 4,
    'PlotArea': 5, 'yAxisLabel': 6, 'xAxisLabel': 7, 'LegendLabel': 8, 'PieLabel': 9,
    'bar': 10, 'pie': 11, 'line': 12, 'pie_slice': 13, 'dot_line': 14
}
categories = [{"id": v, "name": k} for k, v in objects_classes.items()]

# COCO dictionary
coco = {
    "images": [],
    "annotations": [],
    "categories": categories
}
annotation_id = 1
image_id = 1

# Add annotation helper
def add_annotation(category_name, box, image_id):
    global annotation_id
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    # Fix negatives
    if w < 0:
        x += w
        w = abs(w)
    if h < 0:
        y += h
        h = abs(h)
    if w == 0 or h == 0:
        return

    segmentation = [[
        x, y,
        x + w, y,
        x + w, y + h,
        x, y + h
    ]]

    coco["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": objects_classes[category_name],
        "bbox": [x, y, w, h],
        "segmentation": segmentation,
        "iscrowd": 0,
        "area": w * h
    })
    annotation_id += 1

# Process files
for filename in tqdm(sorted(os.listdir(annotation_dir))):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(annotation_dir, filename)
    base_name = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, f"{base_name}.png")

    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        continue

    with open(json_path) as f:
        ann = json.load(f)

    width, height = Image.open(image_path).size
    coco["images"].append({
        "id": image_id,
        "file_name": f"{base_name}.png",
        "width": width,
        "height": height
    })

    gfi = ann.get("general_figure_info", {})
    raw_chart_type = ann.get("type", "")
    label_type = "PieLabel" if raw_chart_type == "pie" else "LegendLabel"

    # Normalize chart_type
    chart_type = {
        "vbar_categorical": "bar",
        "hbar_categorical": "bar",
        "h_bar": "bar",
        "v_bar": "bar",
        "pie": "pie",
        "line": "line",
        "dot_line": "dot_line"
    }.get(raw_chart_type, raw_chart_type)

    if chart_type not in objects_classes:
        print(f"Skipped unknown chart_type '{chart_type}' in file: {filename}")
        continue

    # Chart title
    if "title" in gfi and "bbox" in gfi["title"]:
        add_annotation("ChartTitle", gfi["title"]["bbox"], image_id)

    # Plot area
    if "plot_info" in gfi and "bbox" in gfi["plot_info"]:
        add_annotation("PlotArea", gfi["plot_info"]["bbox"], image_id)

    # Axes
    for axis, axis_title, axis_label in [("x_axis", "xAxisTitle", "xAxisLabel"), ("y_axis", "yAxisTitle", "yAxisLabel")]:
        if axis in gfi:
            axis_data = gfi[axis]
            if "label" in axis_data:
                label_data = axis_data["label"]
                if isinstance(label_data, dict) and "bbox" in label_data:
                    bbox = label_data["bbox"]
                    if all(k in bbox for k in ["x", "y", "w", "h"]):
                        add_annotation(axis_title, bbox, image_id)
            if "major_labels" in axis_data:
                for tick_box in axis_data["major_labels"].get("bboxes", []):
                    if all(k in tick_box for k in ["x", "y", "w", "h"]):
                        add_annotation(axis_label, tick_box, image_id)

    # Legend
    if "legend" in gfi and "items" in gfi["legend"]:
        for item in gfi["legend"]["items"]:
            if "label" in item and "bbox" in item["label"]:
                label_box = item["label"]["bbox"]
                add_annotation("LegendLabel", label_box, image_id)
            if "preview" in item and "bbox" in item["preview"]:
                preview_box = item["preview"]["bbox"]
                add_annotation("LegendPreview", preview_box, image_id)

    # Chart content
    for model in ann.get("models", []):
        for visual_box in model.get("bboxes", []):
            if all(k in visual_box for k in ["x", "y", "w", "h"]):
                add_annotation(chart_type, visual_box, image_id)
        if "text_bbox" in model and all(k in model["text_bbox"] for k in ["x", "y", "w", "h"]):
            text_box = model["text_bbox"]
            add_annotation(label_type, text_box, image_id)

    image_id += 1

# Save
with open(output_path, "w") as f:
    json.dump(coco, f)
print(f"COCO file saved to {output_path}")
