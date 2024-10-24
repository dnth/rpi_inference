import argparse
import time

import cv2
import numpy as np

import onnxruntime


# sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# tanh函数
def tanh(x):
    return 2.0 / (1 + np.exp(-2 * x)) - 1


# 数据预处理
def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255

    return output.astype("float32")


# nms算法
def nms(dets, thresh=0.45):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output


def detection(session, img, input_width, input_height, thresh):
    pred = []

    H, W, _ = img.shape

    data = preprocess(img, [input_width, input_height])

    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]

    feature_map = feature_map.transpose(1, 2, 0)
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score**0.6) * (cls_score**0.4)

            if score > thresh:
                cls_index = np.argmax(data[5:])
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height

                # cx,cy,w,h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    # Add this check before calling nms
    if not pred:
        return []  # Return an empty list if no detections

    return nms(np.array(pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastestDet Webcam Detection")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for detection (default: 0.65)",
    )
    args = parser.parse_args()

    # Load the model with optimizations
    input_width, input_height = 352, 352
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.intra_op_num_threads = 4  # Adjust based on your Pi's CPU cores
    session = onnxruntime.InferenceSession("FastestDet.onnx", sess_options=sess_options)

    # Load label names
    names = []
    with open("coco.names", "r") as f:
        names = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Unable to open camera with index {args.camera}")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform object detection
        start = time.perf_counter()
        bboxes = detection(session, frame, input_width, input_height, args.threshold)
        end = time.perf_counter()
        inference_time = (end - start) * 1000.0

        # Draw bounding boxes and labels
        for b in bboxes:
            obj_score, cls_index = b[4], int(b[5])
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{obj_score:.2f}", (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, names[cls_index], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        # Display inference time
        cv2.putText(
            frame,
            f"Inference Time: {inference_time:.2f}ms",
            (10, 30),
            0,
            0.7,
            (0, 0, 255),
            2,
        )

        # Show the frame
        cv2.imshow("FastestDet Webcam Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
