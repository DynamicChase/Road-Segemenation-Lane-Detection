---

# Road Detection and Lane Segmentation with YOLOv8

This project implements a road detection and lane segmentation system using the YOLOv8 segmentation model. The model has been trained on open-source video datasets, with annotations created using Label Studio. This system is designed for real-time applications in autonomous driving and advanced driver-assistance systems (ADAS).

## Features

- **Real-Time Segmentation**: Accurately detects road areas and lane markings in real-time video streams.
- **High Accuracy**: Utilizes the YOLOv8 architecture for state-of-the-art performance in segmentation tasks.
- **Custom Dataset Support**: Trained on a custom dataset annotated with Label Studio, allowing for flexibility in application.
- **Visualization**: Visualizes detected lanes and road areas directly on video frames for easy interpretation.
- **Easy to Use**: Simple command-line interface for training and inference.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLOv8

You can install the required libraries using pip. Create a `requirements.txt` file with the following content:

```
opencv-python
numpy
torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113  # Adjust according to your CUDA version
ultralytics
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. **Clone the Repository:**

   Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/DynamicChase/Road-Segemenation-Lane-Detection.git
   cd Road-Segemenation-Lane-Detection
   ```

2. **Download YOLO Model:**

   Since the trained YOLO model cannot be uploaded directly to GitHub, you can download it from Google Drive using the following link:

   [Download YOLO Model](https://drive.google.com/file/d/16lWF8XGr7Nx91DtXkRps0Kdx9wDX9yuK/view?usp=sharing)

3. **Prepare Your Dataset:**

   Place your training images in the `data/train` directory and testing images in the `data/test` directory. Ensure that your dataset is properly annotated using Label Studio.

4. **Train the Model (Optional):**

   If you want to train the model on your own dataset, create a YAML configuration file (e.g., `data.yaml`) that includes:

   ```yaml
   train: /path/to/train/images
   val: /path/to/val/images
   nc: 1
   names: ['Road']
   ```

   Then run the training command:

   ```bash
   yolo task=segment mode=train model=yolov8m-seg.pt data=data.yaml imgsz=640 epochs=100 batch=16 name=my_yolov8_model exist_ok=True
   ```

5. **Run Inference:**

   To run inference on a video file, execute the following command:

   ```bash
   python main.py
   ```

   Ensure that your video file path is correctly set in `main.py` (currently set to `/home/sm/Desktop/yolov8/test2.mp4`).

## Code Overview

The main functionality is implemented in `main.py`. Here’s a brief overview of how it works:

1. **Model Loading**: The YOLOv8 segmentation model is loaded from a specified path.
2. **Video Capture**: The script captures video frames from a specified video file.
3. **Segmentation**: For each frame, the model performs segmentation to identify road areas.
4. **Lane Detection**: The script uses functions from `lane.py` to detect lanes on the segmented image.
5. **Visualization**: Detected lanes and road areas are visualized on the original frame.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
