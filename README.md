# YOLO-SAM Pipeline

## Project Overview 🚁
This project integrates YOLOv8s and SAM models to create a powerful object detection, semantic and instance segmentation pipeline. The project aims to optimize these models for high performance and efficiency, achieving results that are on par with state-of-the-art models. The pipeline has been applied to the Cityscapes dataset for segmentation tasks and lane detection using OpenCV.

## Setup and Installation

### Prerequisites
- Python 3.8 or later
- CUDA-enabled GPU (for training and inference)
- [PyTorch](https://pytorch.org/get-started/locally/) (compatible version with your CUDA)
- Other required Python packages (listed in `requirements.txt`)

### Installation Steps ⚒
1. **Clone the Repository**
    ```bash
    git clone https://github.com/DINAMOHMD/SAM-YOLOv8s-Project.git
    cd yolo-sam-segmentation
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Pre-trained Models**
    - Download the YOLOv8s and SAM pre-trained models and place them in the `models` directory. (Provide links or instructions for downloading these models)

## Running the Project 🏃

### Training the YOLOv8s Model
1. **Prepare the Dataset**
    - The dataset should be annotated using Roboflow. You can download the annotated dataset from [[Roboflow here](https://universe.roboflow.com/new-amina/object-detection-new-amina-dina)](#).
    - The original dataset can be found on Google Drive [[here](https://drive.google.com/drive/folders/1Oi5mQ5hfkxmf3suYUi6MPn62V8aONu47?usp=drive_link)](#).

2. **Train the Model**
    ```bash
!yolo task = detect mode=train model=yolov8s.pt data=data.yaml epochs=50 imgsz=800 plots=True iou=0.5 conf=0.25
```

### Performing Segmentation with SAM
1. **Segmentation Inference**
    ```bash
    python segment_sam.py --input data/cityscapes/images --output results/segmentation
    ```

2. **Comparing with Segformer**
    - Use the provided script to compare SAM results with Segformer.
    ```bash
    python compare_segmentation.py --sam results/segmentation --segformer results/segformer
    ```

### Lane Detection Using OpenCV 
1. **Lane Detection Pipeline**
   You can find All files through: https://github.com/Asmaa-Ali7/Lane_Line_detection   ✈ 👀
2. **Methodology**
    - The lane detection script uses OpenCV to process video frames.

## Achieved Results 💎
- **Cityscapes Dataset Annotation**: We used Roboflow to annotate the Cityscapes dataset, preparing it for training and evaluation.
- **YOLOv8s Fine-Tuning**: Utilized a grid search approach to fine-tune the YOLOv8s model for optimal performance.
- **Segmentation with SAM**: Implemented SAM for segmentation tasks and compared its performance to Segformer. SAM demonstrated superior results in terms of accuracy and efficiency.
- **Lane Detection**: Developed a lane detection pipeline using OpenCV, leveraging the strengths of the integrated models for precise and reliable lane detection.


## Additional Notes
- **Model Optimization**: Grid search was used to fine-tune YOLOv8s for optimal performance.
- **Performance Comparison**: The SAM model's segmentation performance was compared to Segformer, showing superior results.
- **Lane Detection**: The pipeline leverages OpenCV for robust lane detection, making it suitable for real-time applications in autonomous driving and driver assistance systems.

## Contributing
We welcome contributions



