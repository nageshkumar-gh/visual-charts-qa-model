from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # or yolov8n.pt to fine-tune

model.train(
    data="/Users/nagesh/Documents/QA_Model/Model/input/yolo_dataset.yaml",

    epochs=50,
    imgsz=640,
    batch=2,
    device="mps"  # for Mac M1/M2
)
