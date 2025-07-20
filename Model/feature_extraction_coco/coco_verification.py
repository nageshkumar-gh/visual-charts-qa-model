import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm

# Paths
annotation_dir = "Data/train/annotations"
image_dir = "Data/train/png"
output_dir = "debug_outputs"
os.makedirs(output_dir, exist_ok=True)

# Class mapping
objects_classes = {
    'Legend': 0, 'yAxisTitle': 1, 'ChartTitle': 2, 'xAxisTitle': 3, 'LegendPreview': 4,
    'PlotArea': 5, 'yAxisLabel': 6, 'xAxisLabel': 7, 'LegendLabel': 8, 'PieLabel': 9,
    'bar': 10, 'pie': 11, 'line': 12, 'pie_slice': 13, 'dot_line': 14
}
id_to_label = {v: k for k, v in objects_classes.items()}

# Helper to draw box
def draw_box(draw, label, bbox, color="red"):
    try:
        x0, y0, w, h = bbox
        # Fix negative width/height
        if w < 0:
            x0 = x0 + w
            w = abs(w)
        if h < 0:
            y0 = y0 + h
            h = abs(h)
        x1, y1 = x0 + w, y0 + h
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, y0), label, fill=color)
    except Exception as e:
        print(f"Error drawing box for {label}: {bbox} - {e}")

# Start processing
for filename in tqdm(sorted(os.listdir(annotation_dir))):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(annotation_dir, filename)
    image_index = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, f"{image_index}.png")
    output_path = os.path.join(output_dir, f"{image_index}_vis.png")

    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        continue

    with open(json_path) as f:
        ann = json.load(f)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    gfi = ann.get("general_figure_info", {})
    chart_type = ann.get("type", "")
    label_type = "PieLabel" if chart_type == "pie" else "LegendLabel"

    if "title" in gfi and "bbox" in gfi["title"]:
        box = gfi["title"]["bbox"]
        draw_box(draw, "ChartTitle", [box["x"], box["y"], box["w"], box["h"]], color="blue")

    if "plot_info" in gfi and "bbox" in gfi["plot_info"]:
        box = gfi["plot_info"]["bbox"]
        draw_box(draw, "PlotArea", [box["x"], box["y"], box["w"], box["h"]], color="gray")

    for axis, axis_title, axis_label in [("x_axis", "xAxisTitle", "xAxisLabel"), ("y_axis", "yAxisTitle", "yAxisLabel")]:
        if axis in gfi:
            axis_data = gfi[axis]
            if "label" in axis_data:
                label_data = axis_data["label"]
                if isinstance(label_data, dict) and "bbox" in label_data:
                    bbox = label_data["bbox"]
                    if all(k in bbox for k in ["x", "y", "w", "h"]):
                        draw_box(draw, axis_title, [bbox["x"], bbox["y"], bbox["w"], bbox["h"]], color="purple")
            if "major_labels" in axis_data:
                for tick_box in axis_data["major_labels"].get("bboxes", []):
                    if all(k in tick_box for k in ["x", "y", "w", "h"]):
                        draw_box(draw, axis_label, [tick_box["x"], tick_box["y"], tick_box["w"], tick_box["h"]], color="purple")

    if "legend" in gfi and "items" in gfi["legend"]:
        for item in gfi["legend"]["items"]:
            if "label" in item and "bbox" in item["label"]:
                label_box = item["label"]["bbox"]
                draw_box(draw, "LegendLabel", [label_box["x"], label_box["y"], label_box["w"], label_box["h"]], color="green")
            if "preview" in item and "bbox" in item["preview"]:
                preview_box = item["preview"]["bbox"]
                draw_box(draw, "LegendPreview", [preview_box["x"], preview_box["y"], preview_box["w"], preview_box["h"]], color="orange")

    for model in ann.get("models", []):
        for visual_box in model.get("bboxes", []):
            if all(k in visual_box for k in ["x", "y", "w", "h"]):
                draw_box(draw, chart_type, [visual_box["x"], visual_box["y"], visual_box["w"], visual_box["h"]], color="red")
        if "text_bbox" in model and all(k in model["text_bbox"] for k in ["x", "y", "w", "h"]):
            text_box = model["text_bbox"]
            draw_box(draw, label_type, [text_box["x"], text_box["y"], text_box["w"], text_box["h"]], color="brown")

    img.save(output_path)
    print(f"Saved: {output_path}")
