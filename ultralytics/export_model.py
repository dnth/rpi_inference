from ultralytics import YOLO

# Load the PyTorch model
model = YOLO("yolov8s.pt")

# Export as an NCNN format
model.export(format="ncnn", imgsz=320, half=True)
