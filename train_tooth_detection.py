import os
import torch
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import yaml

class ToothDetectionTrainer:
    def __init__(self, data_yaml_path, model_name='yolov8s.pt'):
        self.data_yaml_path = data_yaml_path
        self.model_name = model_name
        self.model = None
        self.results = None
        
    def train_model(self, epochs=100, imgsz=640, batch_size=16, project='tooth_detection_runs'):
        print(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)
        
        print("Starting training...")
        self.results = self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project,
            name='tooth_detection_exp',
            save=True,
            plots=True,
            verbose=True
        )
        
        print("Training completed!")
        return self.results
    
    def evaluate_model(self, test_data_path=None):
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return
        
        print("Evaluating model...")
        if test_data_path:
            metrics = self.model.val(data=test_data_path)
        else:
            metrics = self.model.val()
        
        return metrics
    
    def predict_and_visualize(self, image_paths, save_dir='predictions'):
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        for img_path in image_paths:
            print(f"Predicting on {img_path}")
            results = self.model(img_path)
            
            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_dir, f"predicted_{img_name}")
            results[0].save(save_path)
            print(f"Saved prediction to {save_path}")
    
    def generate_confusion_matrix(self, val_data_path=None):
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return
        
        with open(self.data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        class_names = [data_config['names'][i] for i in range(len(data_config['names']))]
        
        metrics = self.evaluate_model(val_data_path)
        
        print("Confusion matrix saved in training results folder")
        print("Check: runs/detect/train/confusion_matrix.png")
        
        return metrics

def load_sample_test_images(test_images_folder, num_samples=5):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    test_images = []
    
    for file in os.listdir(test_images_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(test_images_folder, file))
        
        if len(test_images) >= num_samples:
            break
    
    return test_images

def main():
    DATA_YAML = './data.yaml'
    MODEL_NAME = 'yolov8s.pt'
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    
    trainer = ToothDetectionTrainer(DATA_YAML, MODEL_NAME)
    
    print("="*50)
    print("STARTING TOOTH DETECTION TRAINING")
    print("="*50)
    
    results = trainer.train_model(
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch_size=BATCH_SIZE,
        project='tooth_detection_runs'
    )
    
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    metrics = trainer.evaluate_model()
    
    print("\n" + "="*50)
    print("GENERATING CONFUSION MATRIX")
    print("="*50)
    
    trainer.generate_confusion_matrix()
    
    print("\n" + "="*50)
    print("GENERATING SAMPLE PREDICTIONS")
    print("="*50)
    
    test_images_folder = './datasets/tooth_detection/test/images'
    if os.path.exists(test_images_folder):
        sample_images = load_sample_test_images(test_images_folder, num_samples=5)
        if sample_images:
            trainer.predict_and_visualize(sample_images, 'sample_predictions')
            print(f"Sample predictions saved in 'sample_predictions' folder")
        else:
            print("No test images found for prediction")
    else:
        print(f"Test images folder not found: {test_images_folder}")
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model trained for {EPOCHS} epochs")
    print(f"Results saved in: tooth_detection_runs/")
    print("Key files to check:")
    print("- Training curves: runs/detect/train/results.png")
    print("- Confusion matrix: runs/detect/train/confusion_matrix.png")
    print("- Best weights: runs/detect/train/weights/best.pt")
    print("- Sample predictions: sample_predictions/")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    main()