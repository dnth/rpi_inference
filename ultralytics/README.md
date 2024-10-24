# rpi_inference
Computer vision inference on Raspberry Pi.

## Install

```bash
pip install ultralytics
```

## Usage

Use PyTorch model:
```bash
python inference_video.py --imgsz 320 --model yolov8n.pt
```

Use NCNN model:
```bash
python inference_video.py --imgsz 320 --model yolov8n_ncnn_model
```
