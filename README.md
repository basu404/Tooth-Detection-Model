# Tooth-Detection-Model

**YOLOv8-based system for automated tooth detection using the FDI numbering system.**  
Includes training, evaluation, and prediction scripts with visualization tools and sample results. Designed for dental AI research and clinical imaging tasks.

---

##  Features
- Detects and classifies 32 individual teeth using **FDI numbering**.
- Outputs bounding boxes in **YOLO format** (normalized).
- Includes:
  - Training pipeline (`train_tooth_detection.py`)
  - Evaluation pipeline (`model_evaluation.py`)
  - Prediction+comparison tool (`prediction.py`)
- Provides key metrics: **mAP@50**, **mAP@50–95**, **precision**, **recall**, and confusion matrix.
- Visual outputs: training curves (`results.png`), confusion matrix, and sample annotated images.
- Supports visualization comparing **predicted vs ground truth** bounding boxes.

---




# Clone the repository
git clone https://github.com/basu404/Tooth-Detection-Model.git

cd Tooth-Detection-Model



# Install dependencies
pip install ultralytics torch opencv-python matplotlib seaborn pyyaml

# Train the Model
python train_tooth_detection.py

# Evaluate Performance
python model_evaluation.py

# Prediction
python prediction.py




