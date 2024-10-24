# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

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
                )  # add border
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
            print(self.classes[classid] + ": " + str(round(conf, 3)))
            cv2.putText(
                srcimg,
                self.classes[classid] + ":" + str(round(conf, 3)),
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                thickness=2,
            )

        return srcimg, inference_time

    def detect_folder(self, img_fold, result_path):
        img_fold = Path(img_fold)
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        img_name_list = filter(
            lambda x: str(x).endswith(".png") or str(x).endswith(".jpg"),
            img_fold.iterdir(),
        )
        img_name_list = list(img_name_list)
        print(f"find {len(img_name_list)} images")

        total_time = 0
        for img_path in tqdm(img_name_list):
            img = cv2.imread(str(img_path))

            srcimg, inference_time = net.detect(img)
            total_time += inference_time
            print(f"Inference time for {img_path.name}: {inference_time:.4f} seconds")
            save_path = str(result_path / img_path.name.replace(".png", ".jpg"))
            cv2.imwrite(save_path, srcimg)

        avg_time = total_time / len(img_name_list)
        print(f"\nAverage inference time: {avg_time:.4f} seconds")
        print(f"FPS: {1/avg_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelpath",
        type=str,
        default="onnx_files/picodet_s_320_lcnet_postprocessed.onnx",
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
    parser.add_argument("--img_fold", dest="img_fold", type=str, default="./imgs")
    parser.add_argument(
        "--result_fold", dest="result_fold", type=str, default="./results"
    )
    args = parser.parse_args()

    net = PicoDet(
        args.modelpath,
        args.classfile,
        prob_threshold=args.confThreshold,
        iou_threshold=args.nmsThreshold,
    )

    net.detect_folder(args.img_fold, args.result_fold)
