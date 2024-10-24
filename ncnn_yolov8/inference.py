import time  # Add this import

import cv2
from ncnn.model_zoo import get_model

image = cv2.imread("../ncnn_fastestdet/test_image.jpg")
model = get_model(
    "yolov8s",
    target_size=640,
    prob_threshold=0.5,
    nms_threshold=0.45,
    num_threads=4,
    use_gpu=True,
)

# Add inference time logging
start_time = time.time()
result = model(image)
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")
