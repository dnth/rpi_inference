import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import onnxruntime as ort


class PicoDet:
    def __init__(
        self, model_pb_path, label_path, prob_threshold=0.4, iou_threshold=0.3
    ):
        self.classes = list(map(lambda x: x.strip(), open(label_path, "r").readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(
            1, 1, 3
        )
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        self.input_shape = (
            self.net.get_inputs()[0].shape[2],
            self.net.get_inputs()[0].shape[3],
        )

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype("float32")

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)

        return img, scale_factor

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
                color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
                color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
                j += 1
                lab >>= 3
        color_map = [color_map[i : i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def detect(self, srcimg):
        img, scale_factor = self.resize_image(srcimg)
        img = self._normalize(img)

        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        start_time = time.time()
        outs = self.net.run(
            None,
            {
                self.net.get_inputs()[0].name: blob,
                self.net.get_inputs()[1].name: scale_factor,
            },
        )
        inference_time = time.time() - start_time

        outs = np.array(outs[0])
        expect_boxes = (outs[:, 1] > 0.5) & (outs[:, 0] > -1)
        np_boxes = outs[expect_boxes, :]

        color_list = self.get_color_map_list(self.num_classes)
        clsid2color = {}

        for i in range(np_boxes.shape[0]):
            classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
            xmin, ymin, xmax, ymax = (
                int(np_boxes[i, 2]),
                int(np_boxes[i, 3]),
                int(np_boxes[i, 4]),
                int(np_boxes[i, 5]),
            )

            if classid not in clsid2color:
                clsid2color[classid] = color_list[classid]
            color = tuple(clsid2color[classid])

            cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
            cv2.putText(
                srcimg,
                f"{self.classes[classid]}: {conf:.3f}",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                thickness=2,
            )

        return srcimg, inference_time


def webcam_detection(args):
    # Initialize PicoDet
    net = PicoDet(
        args.modelpath,
        args.classfile,
        prob_threshold=args.confThreshold,
        iou_threshold=args.nmsThreshold,
    )

    # Open webcam
    cap = cv2.VideoCapture(args.camera_index)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run detection
        result_frame, inference_time = net.detect(frame)

        # Display FPS and inference latency
        fps = 1 / inference_time
        cv2.putText(
            result_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result_frame,
            f"Latency: {inference_time*1000:.2f} ms",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show the result
        cv2.imshow("PicoDet Webcam Detection", result_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelpath",
        type=str,
        default="onnx_files/picodet_xs_320_lcnet_postprocessed.onnx",
        help="onnx filepath",
    )
    parser.add_argument(
        "--classfile", type=str, default="coco_label.txt", help="classname filepath"
    )
    parser.add_argument(
        "--confThreshold", default=0.5, type=float, help="class confidence"
    )
    parser.add_argument(
        "--nmsThreshold", default=0.6, type=float, help="nms iou thresh"
    )
    parser.add_argument(
        "--camera_index", type=int, default=0, help="camera device index"
    )
    args = parser.parse_args()

    webcam_detection(args)
