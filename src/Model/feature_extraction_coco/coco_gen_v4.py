import os
import json
from PIL import Image

# Paths
# annotation_dir = "Data/train/annotations"
# image_dir = "Data/train/png"
# output_path = "Data/train/coco_files/train_coco_annotations_v4.json"

# annotation_dir = "Data/val/annotations"
# image_dir = "Data/val/png"
# output_path = "Data/val/coco_files/val_coco_annotations_v4.json"

annotation_dir = "Data/test/annotations"
image_dir = "Data/test/png"
output_path = "Data/test/coco_files/test_coco_annotations_v4.json"

# Class mapping
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
categories = [{"id": v, "name": k} for k, v in objects_classes.items()]

# COCO JSON template
coco = {"images": [], "annotations": [], "categories": categories}
annotation_id = 1
image_id = 1

def add_annotation(label, bbox, image_id):
    global annotation_id
    if bbox["w"] <= 0 or bbox["h"] <= 0:
        return
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    coco["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": objects_classes[label],
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]
    })
    annotation_id += 1

# Process annotation files
for filename in os.listdir(annotation_dir):
    if not filename.endswith(".json"):
        continue
    path = os.path.join(annotation_dir, filename)
    with open(path) as f:
        ann = json.load(f)

    base = filename.replace(".json", "")
    img_file = f"{base}.png"
    image_path = os.path.join(image_dir, img_file)

    
    #width, height = 800, 600  # use real size if available

    try:
      with Image.open(image_path) as img:
        width, height = img.size
    except FileNotFoundError:
      print(f" Image not found: {image_path}, using default size.")

    coco["images"].append({
        "id": image_id,
        "file_name": img_file,
        "width": width,
        "height": height
    })

    chart_type = ann.get("type", "")
    gfi = ann.get("general_figure_info", {})

    # Chart Title
    if "title" in gfi and "bbox" in gfi["title"]:
        add_annotation("ChartTitle", gfi["title"]["bbox"], image_id)

    # Plot Area
    if "figure_info" in gfi and "bbox" in gfi["figure_info"]:
        add_annotation("PlotArea", gfi["figure_info"]["bbox"], image_id)

    # Legend Labels
    for item in gfi.get("legend", {}).get("items", []):
        if "label" in item and "bbox" in item["label"]:
            add_annotation("LegendLabel", item["label"]["bbox"], image_id)

    # X Axis Labels
    for tick in gfi.get("x_axis", {}).get("major_labels", {}).get("bboxes", []):
        add_annotation("xAxisLabel", tick, image_id)

    # Y Axis Labels
    for tick in gfi.get("y_axis", {}).get("major_labels", {}).get("bboxes", []):
        add_annotation("yAxisLabel", tick, image_id)

    # X Axis title
    #for tick in gfi.get("x_axis", {}).get("label", {}).get("bbox", []):
    #    add_annotation("xAxisTitle", tick, image_id)
    
    # Y Axis Labels
    for tick in gfi.get("y_axis", {}).get("label", {}).get("bbox", []):
        add_annotation("yAxisTitle", tick, image_id)

    # Models (data + text)
    for model in ann.get("models", []):
        if chart_type == "pie":
        # Use only text_bbox for pie labels
            if "text_bbox" in model:
                add_annotation("PieLabel", model["text_bbox"], image_id)
        
        elif chart_type == "v_bar":
          for bbox in model.get("bboxes", []):
            add_annotation("v_bar", bbox, image_id)

        elif chart_type == "h_bar":
          for bbox in model.get("bboxes", []):
            add_annotation("h_bar", bbox, image_id)
        
        elif chart_type == "line":
          for bbox in model.get("bboxes", []):
            add_annotation("line", bbox, image_id)
        
        

      #  elif chart_type in ["v_bar", "h_bar" , "line"]:
      #    label = "bar" if chart_type in ["v_bar", "h_bar"] else "line"
       #   for bbox in model.get("bboxes", []):
       #     add_annotation(label, bbox, image_id)

    image_id += 1

# Add dataset metadata 
coco["info"] = {
    "description": "QA_Model",
    "version": "2.0",
    "year": 2025,
    "contributor": "Auto",
    "date_created": "2025-07-16"
}

# Save COCO JSON
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(coco, f, indent=4)
print(f"Saved COCO JSON to: {output_path}")
