from ultralytics import YOLO

# Load the PyTorch model
model = YOLO("yolov8n.pt")

# Export as an NCNN format
model.export(format="ncnn")
