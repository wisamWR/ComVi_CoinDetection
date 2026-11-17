"""Kaggle GPU job: train one YOLO26 detector per currency and evaluate on the held-out test split.

Inputs : Kaggle dataset mehmedwisam/comvi-money-detection (built by tools/build_dataset.py)
Outputs: /kaggle/working/out/<task>/{best.pt, metrics.json, train/...}
"""
import glob
import json
import os
import shutil
import subprocess
import sys

if not os.path.isdir("/kaggle/input"):
    sys.exit("Refusing to run: this job is meant for Kaggle GPU (T4 x2), not a local machine.")

import torch  # noqa: E402

if torch.cuda.device_count() < 2:
    sys.exit(f"Expected 2 GPUs (T4 x2), found {torch.cuda.device_count()}. Push with --accelerator NvidiaTeslaT4.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.213"], check=True)
from ultralytics import YOLO  # noqa: E402

DEVICE = [0, 1]  # DDP across both T4s

TASKS = {
    # task: (model, epochs, imgsz, total batch split over both GPUs)
    "usd_coins": ("yolo26s.pt", 100, 640, 32),
    "idr_notes": ("yolo26s.pt", 100, 640, 32),
}
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "out")


def find_task_dir(task):
    hits = glob.glob(f"/kaggle/input/**/{task}/data.yaml", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{task}/data.yaml not found under /kaggle/input")
    return os.path.dirname(hits[0])


def stage(task):
    """Copy to a writable dir (label caches) and point data.yaml at it."""
    dst = os.path.join("/tmp/data", task)
    if not os.path.isdir(dst):
        shutil.copytree(find_task_dir(task), dst)
    yaml_path = os.path.join(dst, "data_abs.yaml")
    with open(os.path.join(dst, "data.yaml")) as f:
        text = f.read().replace("path: .", f"path: {dst}")
    with open(yaml_path, "w") as f:
        f.write(text)
    return yaml_path


def run(task, weights, epochs, imgsz, batch):
    data = stage(task)
    model = YOLO(weights)
    # No RAM cache: under DDP each rank would hold its own copy (~2x 10 GB for idr_notes).
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, patience=20, device=DEVICE,
                cache=False, workers=4, seed=0, project=os.path.join(OUT, task), name="train",
                exist_ok=True, plots=True)
    best = os.path.join(OUT, task, "train", "weights", "best.pt")
    test = YOLO(best).val(data=data, split="test", imgsz=imgsz, batch=batch, device=0,
                          project=os.path.join(OUT, task), name="test", exist_ok=True)
    names = test.names
    metrics = {
        "task": task, "weights": weights, "epochs": epochs, "imgsz": imgsz,
        "test": {"mAP50": test.box.map50, "mAP50-95": test.box.map,
                 "precision": test.box.mp, "recall": test.box.mr},
        "per_class_mAP50-95": {names[i]: float(m) for i, m in zip(test.box.ap_class_index, test.box.maps[test.box.ap_class_index])},
        "speed_ms": test.speed,
    }
    shutil.copy(best, os.path.join(OUT, task, "best.pt"))
    with open(os.path.join(OUT, task, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    for task, cfg in TASKS.items():
        run(task, *cfg)
