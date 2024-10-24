import argparse
import time

import cv2
from ncnn.model_zoo import get_model

# Class names
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Parse command line arguments
parser = argparse.ArgumentParser(description="YOLOv8 webcam inference")
parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
args = parser.parse_args()

# Initialize the webcam
cap = cv2.VideoCapture(args.camera)

# Initialize the YOLOv8 model
model = get_model(
    "yolov8s",
    target_size=640,
    prob_threshold=0.5,
    nms_threshold=0.45,
    num_threads=4,
    use_gpu=True,
)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    start_time = time.time()
    result = model(frame)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Draw bounding boxes and labels
    for detection in result:
        rect = detection.rect
        x1, y1 = int(rect.x), int(rect.y)
        x2, y2 = int(rect.x + rect.w), int(rect.y + rect.h)

        label_id = int(detection.label)
        label = (
            class_names[label_id]
            if label_id < len(class_names)
            else f"Unknown ({label_id})"
        )
        prob = detection.prob

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}: {prob:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Display inference time on the frame
    cv2.putText(
        frame,
        f"Inference time: {inference_time:.2f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # Display the resulting frame
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
