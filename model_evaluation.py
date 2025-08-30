import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import yaml
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model_path, data_yaml_path):
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.model = YOLO(model_path)
        
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        self.class_names = [self.data_config['names'][i] for i in range(len(self.data_config['names']))]
    
    def evaluate_test_set(self):
        print("Evaluating model on test set...")
        
        metrics = self.model.val(
            data=self.data_yaml_path,
            split='test',
            save_json=True,
            save_hybrid=True,
            plots=True
        )
        
        print(f"Test Results:")
        print(f"mAP@50: {metrics.box.map50:.4f}")
        print(f"mAP@50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def generate_sample_predictions(self, test_images_folder, num_samples=5, save_dir='final_predictions'):
        os.makedirs(save_dir, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        test_images = []
        
        for file in os.listdir(test_images_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                test_images.append(os.path.join(test_images_folder, file))
            
            if len(test_images) >= num_samples:
                break
        
        print(f"Generating predictions for {len(test_images)} sample images...")
        
        for i, img_path in enumerate(test_images):
            print(f"Processing {img_path}")
            
            results = self.model(img_path, conf=0.25, iou=0.5)
            
            img_name = f"sample_{i+1}_{os.path.basename(img_path)}"
            save_path = os.path.join(save_dir, img_name)
            
            annotated_img = results[0].plot(
                conf=True,
                labels=True,
                boxes=True,
                line_width=2
            )
            
            cv2.imwrite(save_path, annotated_img)
            print(f"Saved prediction to: {save_path}")
            
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                print(f"  - Found {len(detections)} teeth")
                for j, (cls, conf) in enumerate(zip(detections.cls, detections.conf)):
                    tooth_name = self.class_names[int(cls)]
                    print(f"    {j+1}. {tooth_name} (confidence: {conf:.3f})")
            else:
                print("  - No teeth detected")
            print()
    
    def create_performance_summary(self, metrics, save_path='model_performance_summary.txt'):
        with open(save_path, 'w') as f:
            f.write("TOOTH DETECTION MODEL - PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Dataset: {self.data_yaml_path}\n")
            f.write(f"Number of Classes: {len(self.class_names)}\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS:\n")
            f.write(f"mAP@50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)\n")
            f.write(f"mAP@50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)\n")
            f.write(f"Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.1f}%)\n")
            f.write(f"Recall: {metrics.box.mr:.4f} ({metrics.box.mr*100:.1f}%)\n\n")
            
            f.write("CLASS NAMES:\n")
            for i, name in enumerate(self.class_names):
                f.write(f"{i:2d}: {name}\n")
        
        print(f"Performance summary saved to: {save_path}")
    
    def copy_important_files(self, destination_folder='submission_files'):
        os.makedirs(destination_folder, exist_ok=True)
        
        import shutil
        
        files_to_copy = [
            ('best.pt', 'Model weights'),
            ('data.yaml', 'Dataset configuration'),
            ('train_tooth_detection.py', 'Training script'),
        ]
        
        runs_path = Path('tooth_detection_runs/tooth_detection_exp')
        if runs_path.exists():
            confusion_matrix_path = runs_path / 'confusion_matrix.png'
            if confusion_matrix_path.exists():
                shutil.copy2(confusion_matrix_path, destination_folder)
                print(f"Copied: confusion_matrix.png")
            
            results_path = runs_path / 'results.png'
            if results_path.exists():
                shutil.copy2(results_path, destination_folder)
                print(f"Copied: results.png (training curves)")
        
        print(f"Important files copied to: {destination_folder}/")

def main():
    BEST_MODEL_PATH = './tooth_detection_runs/tooth_detection_exp/weights/best.pt'
    DATA_YAML_PATH = './data.yaml'
    TEST_IMAGES_FOLDER = './datasets/tooth_detection/test/images'
    
    print("=" * 60)
    print("FINAL MODEL EVALUATION & RESULTS GENERATION")
    print("=" * 60)
    
    evaluator = ModelEvaluator(BEST_MODEL_PATH, DATA_YAML_PATH)
    
    print("\n1. EVALUATING ON TEST SET...")
    metrics = evaluator.evaluate_test_set()
    
    print("\n2. GENERATING SAMPLE PREDICTIONS...")
    if os.path.exists(TEST_IMAGES_FOLDER):
        evaluator.generate_sample_predictions(
            TEST_IMAGES_FOLDER, 
            num_samples=5,
            save_dir='final_sample_predictions'
        )
    else:
        print(f"Test images folder not found: {TEST_IMAGES_FOLDER}")
    
    print("\n3. CREATING PERFORMANCE SUMMARY...")
    evaluator.create_performance_summary(metrics, 'final_model_performance.txt')
    
    print("\n4. ORGANIZING SUBMISSION FILES...")
    evaluator.copy_important_files('submission_ready')
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("Files ready for submission:")
    print("- final_sample_predictions/ (sample result images)")
    print("- final_model_performance.txt (metrics summary)")
    print("- submission_ready/ (all important files)")
    print("- confusion_matrix.png (if available)")
    print("- results.png (training curves)")

if __name__ == "__main__":
    main()