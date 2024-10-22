import argparse
import time

import cv2
from ultralytics import YOLO

# Add argument parser
parser = argparse.ArgumentParser(description="Run YOLOv8 inference on webcam feed")
parser.add_argument("--imgsz", type=int, default=320, help="Image size for inference")
parser.add_argument(
    "--model", type=str, default="yolov8n.pt", help="Path to YOLO model"
)
parser.add_argument(
    "--source", type=int, default=0, help="Webcam source (default is 0)"
)
parser.add_argument(
    "--output", type=str, default=None, help="Path to save output video (optional)"
)
args = parser.parse_args()

# Load the YOLO model
model = YOLO(args.model)

# Open the webcam
cap = cv2.VideoCapture(args.source)

# Initialize video writer
out = None
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))


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

    # Write the frame to the output video if specified
    if out:
        out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
