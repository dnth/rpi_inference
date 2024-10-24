import math
import multiprocessing
import queue
import time

import cv2
import numpy as np

import ncnn


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x):
    return 2.0 / (1.0 + math.exp(-2 * x)) - 1


class TargetBox:
    def __init__(self, x1, y1, x2, y2, category, score):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.category = category
        self.score = score

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def intersection_area(a, b):
    if a.x1 > b.x2 or a.x2 < b.x1 or a.y1 > b.y2 or a.y2 < b.y1:
        return 0.0
    inter_width = min(a.x2, b.x2) - max(a.x1, b.x1)
    inter_height = min(a.y2, b.y2) - max(a.y1, b.y1)
    return inter_width * inter_height


def nms_handle(src_boxes, iou_threshold=0.45):
    sorted_boxes = sorted(src_boxes, key=lambda x: x.score, reverse=True)
    dst_boxes = []

    for i in range(len(sorted_boxes)):
        keep = True
        for j in range(len(dst_boxes)):
            if sorted_boxes[i].category == dst_boxes[j].category:
                inter_area = intersection_area(sorted_boxes[i], dst_boxes[j])
                union_area = sorted_boxes[i].area() + dst_boxes[j].area() - inter_area
                iou = inter_area / union_area
                if iou > iou_threshold:
                    keep = False
                    break
        if keep:
            dst_boxes.append(sorted_boxes[i])

    return dst_boxes


def detect_objects(img, net, class_names, thresh=0.65):
    img_width, img_height = img.shape[1], img.shape[0]

    # Preprocess image
    input_width, input_height = 352, 352
    mat_in = ncnn.Mat.from_pixels_resize(
        img,
        ncnn.Mat.PixelType.PIXEL_BGR,
        img_width,
        img_height,
        input_width,
        input_height,
    )

    # Normalize
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    # Create extractor
    ex = net.create_extractor()

    # Set input tensor
    ex.input("input.1", mat_in)

    # Get output tensor
    ret, output = ex.extract("758")

    # Process detections
    target_boxes = []
    class_num = len(class_names)

    for h in range(output.h):
        for w in range(output.w):
            obj_score_index = h * output.w + w
            obj_score = output[obj_score_index]

            max_score = 0.0
            category = -1
            for i in range(class_num):
                cls_score_index = ((5 + i) * output.h * output.w) + (h * output.w) + w
                cls_score = output[cls_score_index]
                if cls_score > max_score:
                    max_score = cls_score
                    category = i

            score = pow(max_score, 0.4) * pow(obj_score, 0.6)

            if score > thresh:
                x_offset_index = (1 * output.h * output.w) + (h * output.w) + w
                y_offset_index = (2 * output.h * output.w) + (h * output.w) + w
                box_width_index = (3 * output.h * output.w) + (h * output.w) + w
                box_height_index = (4 * output.h * output.w) + (h * output.w) + w

                x_offset = tanh(output[x_offset_index])
                y_offset = tanh(output[y_offset_index])
                box_width = sigmoid(output[box_width_index])
                box_height = sigmoid(output[box_height_index])

                cx = (w + x_offset) / output.w
                cy = (h + y_offset) / output.h

                x1 = int((cx - box_width * 0.5) * img_width)
                y1 = int((cy - box_height * 0.5) * img_height)
                x2 = int((cx + box_width * 0.5) * img_width)
                y2 = int((cy + box_height * 0.5) * img_height)

                target_boxes.append(TargetBox(x1, y1, x2, y2, category, score))

    # Apply NMS
    nms_boxes = nms_handle(target_boxes)

    return nms_boxes


def process_frame(frame_queue, result_queue, net, class_names):
    while True:
        try:
            frame, frame_number = frame_queue.get(timeout=1)
            results = detect_objects(frame, net, class_names)
            result_queue.put((frame, results, frame_number))
        except queue.Empty:
            break


def main():
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

    # Load NCNN model with proper thread setting
    net = ncnn.Net()
    net.opt.num_threads = 4  # Set the desired number of threads
    net.load_param("FastestDet.param")
    net.load_model("FastestDet.bin")
    print("NCNN model loaded successfully...")

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create queues for multiprocessing
    frame_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Determine the number of cores to use (leave one core free for the main process)
    num_cores = multiprocessing.cpu_count() - 1

    # Create and start worker processes
    processes = []
    for _ in range(num_cores):
        p = multiprocessing.Process(
            target=process_frame, args=(frame_queue, result_queue, net, class_names)
        )
        p.start()
        processes.append(p)

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Add frame to the queue for processing
        frame_queue.put((frame.copy(), frame_number))

        # Process results
        try:
            processed_frame, results, processed_frame_number = result_queue.get(
                timeout=1 / 30
            )  # Adjust timeout as needed

            # Draw results on the processed frame
            for box in results:
                cv2.rectangle(
                    processed_frame, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2
                )
                cv2.putText(
                    processed_frame,
                    f"{class_names[box.category]} {box.score:.2f}",
                    (box.x1, box.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            end_time = time.time()
            latency = (end_time - start_time) * 1000
            fps = 1 / (end_time - start_time)

            # Add FPS and latency text
            cv2.putText(
                processed_frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                processed_frame,
                f"Latency: {latency:.2f} ms",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display the frame
            cv2.imshow("FastestDet Webcam", processed_frame)

        except queue.Empty:
            pass

        frame_number += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Signal worker processes to stop
    for _ in range(num_cores):
        frame_queue.put(None)

    # Wait for worker processes to finish
    for p in processes:
        p.join()

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
