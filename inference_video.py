import argparse
import time

import cv2
from ultralytics import YOLO

# Add argument parser
parser = argparse.ArgumentParser(description="Run YOLOv8 inference on webcam feed")
parser.add_argument("--imgsz", type=int, default=320, help="Image size for inference")
args = parser.parse_args()

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


prev_time = time.time()
fps = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame, imgsz=args.imgsz)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Calculate and draw FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    draw_fps(annotated_frame, fps)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
