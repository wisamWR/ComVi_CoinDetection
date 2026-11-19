# ComVi Money Detection

Detect money in photos, video or a live camera, and count the total value. One YOLO26 detector per currency:

| Model | Detects | Classes |
|---|---|---|
| `models/idr_notes.pt` | Indonesian Rupiah banknotes | Rp 1.000, 2.000, 5.000, 10.000, 20.000, 50.000, 100.000 |
| `models/usd_coins.pt` | US coins | penny, nickel, dime, quarter |

Each currency has its own model. Coins and notes look very different, and separate models stay smaller and can be retrained on their own.

## Results

Measured on a held-out **test** split that was never used for training or model selection. Splits are grouped by the original photo or video, so near-duplicate images never appear in two splits.

| Model | Test images | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| IDR notes | 326 | 98.6% | 92.4% | 99.0% | 98.4% |
| USD coins | 74 | 97.8% | 89.1% | 96.9% | 98.3% |

Per-class mAP50-95:

| IDR | 1k | 2k | 5k | 10k | 20k | 50k | 100k |
|---|---|---|---|---|---|---|---|
| | 92.1 | 91.3 | 94.4 | 93.9 | 88.9 | 90.8 | 95.5 |

| USD | penny | nickel | dime | quarter |
|---|---|---|---|---|
| | 91.0 | 86.9 | 88.2 | 90.5 |

Inference is about 10 ms per image on a Tesla T4 at 640 px. Full metrics are in `models/*.metrics.json`.

**Known limitations**
- The first IDR evaluation reported 82.4% precision and 86.7% mAP50-95. One source (`rupiah-emisi-2022`) also labels parts of a note (logo, portrait) with the note's class, so correct single-note predictions were scored as misses. `tools/build_dataset.py` now drops boxes nested inside a same-class box; the numbers above are the same model scored on the cleaned test labels.
- The IDR test set comes mostly from two Roboflow sources. Photos from other cameras, lighting or backgrounds may score lower.
- The IDR model covers banknotes only. Rupiah coins are not detected yet.
- The v1 README reported 94.0% mAP50-95 for the coin model. That figure came from the validation set used during training. The 89.1% above is from a separate test split, so it is a fairer estimate.

## Install

Python 3.10+.

```bash
git clone https://github.com/wisamWR/ComVi_CoinDetection.git
cd ComVi_CoinDetection
pip install -r requirements.txt
```

## Web app (Streamlit)

```bash
streamlit run app.py
```

Pick a currency in the sidebar, then upload a photo or use the camera. The app shows the detected denominations and the total value.

## Command line

```bash
# Webcam, Rupiah
python money_detect.py --currency idr --source usb0 --resolution 1280x720

# Single image or a folder of images, US coins
python money_detect.py --currency usd --source coins.jpg
python money_detect.py --currency usd --source ./photos/

# Video, with a stricter threshold, recorded to demo1.avi
python money_detect.py --currency idr --source notes.mp4 --thresh 0.6 --resolution 1280x720 --record
```

| Argument | Description | Default |
|---|---|---|
| `--currency` | `idr` or `usd` | required |
| `--source` | image, folder, video, `usb0`, `picamera0` | required |
| `--thresh` | minimum confidence | `0.5` |
| `--resolution` | display resolution `WxH` | source resolution |
| `--record` | record video/camera output to `demo1.avi` | off |
| `--model` | custom weights path | `models/<currency>.pt` |
| `--device` | `cpu`, `0`, ... | auto |

Keys: **Q** quit, **S** pause, **P** save `capture.png`.

## Using it from Python

```python
import cv2
from money_detection import MoneyDetector

detector = MoneyDetector("idr")
frame = cv2.imread("wallet.jpg")
dets = detector.detect(frame, conf=0.5)
cur = detector.currency
print(cur.format(cur.total(d.cls for d in dets)))   # e.g. "Rp 175.000"
```

## Project structure

```
ComVi_CoinDetection/
├── app.py                    # Streamlit web app
├── money_detect.py           # CLI: image / folder / video / camera
├── money_detection/          # detector wrapper + currency definitions
├── models/                   # trained weights + test metrics
├── tools/
│   ├── download_roboflow.py  # fetch source datasets (needs ROBOFLOW_API_KEY)
│   ├── build_dataset.py      # merge, de-duplicate, leak-free split
│   └── configs/              # one YAML per dataset
├── kaggle/train/             # Kaggle GPU training job (T4 x2)
├── tests/
└── archive/                  # v1 Colab notebook (YOLOv11, US coins only)
```

## Reproducing the models

1. **Get the source data.**
   - Rupiah: run `python tools/download_roboflow.py data/raw` with `ROBOFLOW_API_KEY` set.
   - US coins: use the original dataset ([Google Drive](https://drive.google.com/drive/folders/1c_2RwK_o2stlLlsr9VmHqMolFCFWW-yj?usp=sharing)), placed under `data/raw/usd-coins/`.
2. **Build the datasets:**
   ```bash
   python tools/build_dataset.py --config tools/configs/idr_notes.yaml
   python tools/build_dataset.py --config tools/configs/usd_coins.yaml
   ```
   The builder does three things:
   - It merges all sources and finds near-duplicate photos with a perceptual hash (dHash). Many Roboflow projects re-upload the same images.
   - It keeps every group of duplicates inside one split, and keeps one image per group in val and test.
   - It caps each original photo or video at 30 images in train.
3. **Train on Kaggle.** Upload `data/processed` as a Kaggle dataset, then run:
   ```bash
   kaggle kernels push -p kaggle/train --accelerator NvidiaTeslaT4
   ```
   The job trains with DDP on both T4 GPUs and writes `best.pt` and `metrics.json` for each currency.

Training setup: YOLO26s, 640 px, batch 32, up to 100 epochs with early stopping (patience 20).

| Model | Epochs run | Best epoch | Training time |
|---|---|---|---|
| USD coins | 100 | 80 | 12 min |
| IDR notes | 84 | 64 | 3 h |

## Data sources

Rupiah banknote data comes from these Roboflow Universe projects, merged and de-duplicated:

| Project | Author | License |
|---|---|---|
| rupiah-emisi-2022 | dimas | CC BY 4.0 |
| indonesia-rupiah-detection | | CC BY 4.0 |
| uang-rupiah-baru | 4IA17 | Public Domain |
| uang-rupiah | project-lca4e | CC BY 4.0 |
| uang-rupiah-kertas-dan-koin | Adelia Clarissa | CC BY 4.0 |

The US coin data is the original ComVi dataset (750 images).

## License

[AGPL-3.0](LICENSE). The project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is AGPL-3.0, so the code and trained weights are distributed under the same license. v1 of this repo was MIT.

## Author

Mohammad Wisam Wiraghina

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26
- [EdjeElectronics](https://github.com/EdjeElectronics) for the original detection script that `money_detect.py` grew from
- The Roboflow Universe dataset authors listed above
