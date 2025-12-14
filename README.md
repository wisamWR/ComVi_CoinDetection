# Coin Detection with YOLOv11

A real-time coin detection system using YOLOv11 object detection model. This project can detect and classify four types of US coins: penny, nickel, dime, and quarter.

## 🎯 Features

- Real-time coin detection from webcam feed
- Support for image, video, and USB camera sources
- Custom-trained YOLOv11 model with 93.9% mAP50-95
- Detects 4 coin types: penny, nickel, dime, quarter
- Video recording capability
- Configurable confidence threshold and resolution

## 📋 Prerequisites

- Python 3.8 or higher
- Conda (Anaconda or Miniconda)
- USB webcam (for real-time detection)

## 🚀 Installation

1. **Clone or download this repository**
   ```bash
   git clone <your-repo-url>
   cd ComVi_CoinDetection
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n yolo-env1 python=3.10
   conda activate yolo-env1
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python numpy
   ```
4. **Download Dataset**
   ```bash
   https://drive.google.com/drive/folders/1c_2RwK_o2stlLlsr9VmHqMolFCFWW-yj?usp=sharing
   ```
## 📁 Project Structure

```
ComVi_CoinDetection/
├── my_model/
│   ├── my_model.pt          # Trained YOLOv11 weights
│   └── train/               # Training results and metrics
├── yolo_detect.py           # Detection script
└── README.md
```

## 🎮 Usage

### Basic Usage (Webcam)

```bash
conda activate yolo-env1
cd C:\Users\wisam\Documents\ComVi_CoinDetection\my_model
curl -o yolo_detect.py https://www.ejtech.io/code/yolo_detect.py
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

### Command Line Arguments

| Argument | Description | Example | Required |
|----------|-------------|---------|----------|
| `--model` | Path to YOLO model file | `my_model.pt` | Yes |
| `--source` | Input source | `usb0`, `test.jpg`, `video.mp4` | Yes |
| `--thresh` | Confidence threshold | `0.5` (default) | No |
| `--resolution` | Display resolution | `1280x720` | No |
| `--record` | Record output video | Flag only | No |

### Usage Examples

**1. Webcam detection with custom resolution:**
```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

**2. Detect coins in a single image:**
```bash
python yolo_detect.py --model my_model.pt --source coin_image.jpg
```

**3. Process video file:**
```bash
python yolo_detect.py --model my_model.pt --source coins_video.mp4 --resolution 640x480
```

**4. Detect from folder of images:**
```bash
python yolo_detect.py --model my_model.pt --source ./test_images/
```

**5. Record webcam detection:**
```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720 --record
```

**6. Custom confidence threshold:**
```bash
python yolo_detect.py --model my_model.pt --source usb0 --thresh 0.7
```

## ⌨️ Keyboard Controls

While running detection:
- **Q**: Quit the program
- **S**: Pause/resume inference
- **P**: Save screenshot as `capture.png`

## 📊 Model Performance

The trained YOLOv11s model achieves:
- **Overall mAP50-95**: 94.0%
- **Overall mAP50**: 99.3%

Per-class performance:
| Coin | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| Penny | 99.9% | 100% | 99.5% | 95.3% |
| Nickel | 96.3% | 98.5% | 99.0% | 91.0% |
| Dime | 99.6% | 98.6% | 99.5% | 95.1% |
| Quarter | 97.1% | 96.3% | 99.3% | 94.4% |

## 🏋️ Model Training

The model was trained on Google Colab with:
- **Base model**: YOLOv11s
- **Dataset**: 750 images (675 train, 75 validation)
- **Epochs**: 60
- **Image size**: 640x640
- **Hardware**: Tesla T4 GPU

Training notebook: `Coin_Detection_Models.ipynb`

## 🛠️ Troubleshooting

**Camera not detected:**
- Check if camera is properly connected
- Try different USB ports (usb0, usb1, etc.)
- Ensure no other application is using the camera

**Low FPS:**
- Reduce resolution: `--resolution 640x480`
- Close other applications
- Ensure GPU drivers are updated

**Model not found:**
- Verify you're in the correct directory
- Check model path: `--model path/to/my_model.pt`

## 📝 Notes

- First run will download required model files
- Webcam detection shows real-time FPS counter
- Recorded videos are saved as `demo1.avi`
- Screenshots are saved as `capture.png`

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Mohammad Wisam Wiraghina

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [EdjeElectronics](https://github.com/EdjeElectronics) for training utilities
