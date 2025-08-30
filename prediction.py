import os
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, data_yaml_path):
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.model = YOLO(model_path)
        
        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        self.class_names = [self.data_config['names'][i] for i in range(len(self.data_config['names']))]

        self.fdi_to_class = {}
        for cls_id, name in self.data_config["names"].items():
            if "(" in name and ")" in name:
                fdi_num = name.split("(")[-1].replace(")", "")
                self.fdi_to_class[fdi_num] = cls_id
    
    def find_tooth_coordinates(self, image_path, tooth_number):
        tooth_number = str(tooth_number)
        
        if tooth_number not in self.fdi_to_class:
            print(f"Tooth {tooth_number} not found in dataset mapping!")
            return None
        
        target_class = self.fdi_to_class[tooth_number]
        
        img = cv2.imread(image_path)
        H, W, _ = img.shape
        
        results = self.model(image_path, conf=0.25)
        detections = results[0].boxes
        
        found_coords = []
        if detections is not None and len(detections) > 0:
            for box, cls, conf in zip(detections.xyxy, detections.cls, detections.conf):
                if int(cls) == int(target_class):
                    x1, y1, x2, y2 = map(float, box.tolist())
                    
                    x_center = ((x1 + x2) / 2) / W
                    y_center = ((y1 + y2) / 2) / H
                    box_w = (x2 - x1) / W
                    box_h = (y2 - y1) / H

                    found_coords.append({
                        "class_id": int(cls),
                        "tooth": tooth_number,
                        "confidence": float(conf),
                        "yolo_format": f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}",
                        "xyxy": [x1, y1, x2, y2]
                    })
        
        return found_coords

    def get_ground_truth(self, label_path, target_class, img_shape):
        H, W, _ = img_shape
        gt_boxes = []

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_path}")
            return gt_boxes

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_c, y_c, w, h = map(float, parts)
                if int(cls_id) == int(target_class):
                    x1 = int((x_c - w/2) * W)
                    y1 = int((y_c - h/2) * H)
                    x2 = int((x_c + w/2) * W)
                    y2 = int((y_c + h/2) * H)
                    gt_boxes.append([x1, y1, x2, y2])
        return gt_boxes
    
    def compare_prediction_with_ground_truth(self, image_path, label_path, tooth_number):
        coords = self.find_tooth_coordinates(image_path, tooth_number)
        if not coords:
            print(f"No predictions for tooth {tooth_number}")
            return

        img = cv2.imread(image_path)
        H, W, _ = img.shape

        target_class = self.fdi_to_class[str(tooth_number)]
        gt_boxes = self.get_ground_truth(label_path, target_class, img.shape)

        for c in coords:
            x1, y1, x2, y2 = map(int, c["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Pred {c['tooth']} ({c['confidence']:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0, 255, 0), 2)

        for gt in gt_boxes:
            x1, y1, x2, y2 = gt
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"GT {tooth_number}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        save_path = f"compare_tooth_{tooth_number}.jpg"
        cv2.imwrite(save_path, img)
        print(f"Saved comparison image: {save_path}")


def main():
    BEST_MODEL_PATH = './tooth_detection_runs/tooth_detection_exp/weights/best.pt'
    DATA_YAML_PATH = './data.yaml'
    SAMPLE_IMAGE = r"C:\Users\ahana\OneDrive\Documents\ToothNumber_TaskDataset\datasets\tooth_detection\test\images\cate6-00063_jpg.rf.ac7f40e1228de6c900c569a5d9f8b044.jpg"
    LABELS_FOLDER = r"C:\Users\ahana\OneDrive\Documents\ToothNumber_TaskDataset\datasets\tooth_detection\test\labels"

    print("=" * 60)
    print("TOOTH DETECTION - COMPARE PREDICTION VS GROUND TRUTH")
    print("=" * 60)
    
    evaluator = ModelEvaluator(BEST_MODEL_PATH, DATA_YAML_PATH)
    
    tooth_number = input("Enter the FDI tooth number (e.g., 11, 36, 48): ")

    img_name = os.path.splitext(os.path.basename(SAMPLE_IMAGE))[0]
    label_path = os.path.join(LABELS_FOLDER, img_name + ".txt")
    
    if not os.path.exists(SAMPLE_IMAGE):
        print(f"Sample image not found: {SAMPLE_IMAGE}")
        return
    
    evaluator.compare_prediction_with_ground_truth(SAMPLE_IMAGE, label_path, tooth_number)


if __name__ == "__main__":
    main()