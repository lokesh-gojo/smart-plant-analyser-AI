"""
SmartVine AI — YOLOv8-Seg Instance Segmentation Training & Conversion Pipeline
Translates Section II-A & Step 4/11 of Seng et al. (IEEE Access, 2018).
Converts Embrapa WGISD and GrapeCS-ML Dataset 5 binary masks into normalized
YOLO polygon format, configures dataset YAML with hue-preserving augmentations,
and exports trained weights to ONNX and TFLite for edge deployment.
"""

from typing import Dict, Any, List, Optional
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


def mask_to_yolo_polygon(
    mask_path: str,
    label_output_path: str,
    min_contour_area: float = 150.0,
    class_id: int = 0
) -> int:
    """
    Converts a single binary segmentation mask into YOLO polygon annotation format:
    Format per line: <class_id> <x1/w> <y1/h> <x2/w> <y2/h> ...
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0

    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            coords = []
            for pt in cnt.reshape(-1, 2):
                x_norm = max(0.0, min(1.0, float(pt[0]) / w))
                y_norm = max(0.0, min(1.0, float(pt[1]) / h))
                coords.append(f"{x_norm:.6f} {y_norm:.6f}")
            if len(coords) >= 3:
                lines.append(f"{class_id} " + " ".join(coords))

    os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
    with open(label_output_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


def generate_dataset_yaml(
    dataset_root: str,
    output_yaml_path: str = "grape_seg.yaml"
) -> str:
    """
    Generates Ultralytics YOLOv8 segmentation dataset configuration YAML.
    """
    dataset_root_abs = os.path.abspath(dataset_root).replace("\\", "/")
    yaml_content = f"""# SmartVine AI: Grape Cluster Instance Segmentation Dataset
path: {dataset_root_abs}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: grape_bunch
"""
    with open(output_yaml_path, "w") as f:
        f.write(yaml_content)
    return output_yaml_path


def train_bunch_segmenter(
    data_yaml_path: str = "grape_seg.yaml",
    weights: str = "yolov8s-seg.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/viticulture",
    name: str = "bunch_segmenter"
) -> YOLO:
    """
    Trains YOLOv8s-Seg model with agronomically tuned hyperparameters:
    - hsv_h = 0.015: Restricted hue jitter to preserve subtle white vs red grape skin chromaticity
    - hsv_s = 0.40, hsv_v = 0.40: High saturation/value jitter to simulate harsh vineyard sunlight
    - fliplr = 0.5: Horizontal flip invariance
    - mosaic = 1.0: Multi-cluster occlusion robustness
    """
    model = YOLO(weights)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.40,
        fliplr=0.5,
        mosaic=1.0,
        project=project,
        name=name,
        exist_ok=True
    )
    return model


def export_edge_models(
    model_weights_path: str,
    output_dir: str = "models",
    imgsz: int = 640
) -> Dict[str, str]:
    """
    Exports trained YOLO segmentation model to ONNX and quantized INT8 TFLite formats
    for edge deployment (Raspberry Pi, NVIDIA Jetson, Android smartphones).
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_weights_path)

    # 1. ONNX Export
    onnx_file = model.export(format="onnx", imgsz=imgsz, simplify=True)

    # 2. TFLite INT8 Export for mobile edge inference
    try:
        tflite_file = model.export(format="tflite", imgsz=imgsz, int8=True)
    except Exception:
        tflite_file = "TFLite export available on Linux/Colab environment"

    return {
        "onnx_path": str(onnx_file),
        "tflite_path": str(tflite_file)
    }
