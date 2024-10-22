from ultralytics import YOLO

model = YOLO("yolov8n.pt")

result = model.predict("https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/00aa2580828a9009.jpg", show=True, save=True)

print(result)

