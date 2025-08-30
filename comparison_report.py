import os
import shutil
import hashlib
from pathlib import Path
from ultralytics import YOLO

# ================== USER CONFIG ==================
MODEL1_PATH = r"runs/detect/train/weights/best.pt"
MODEL2_PATH = r"tooth_detection_runs/tooth_detection_exp/weights/best.pt"
DATA_YAML   = r"data.yaml"
SUBMISSION_FOLDER = "submission_ready"
# =================================================

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def evaluate(model_path):
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, split="test", plots=False)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

def better(m1, m2):
    # Decide winner with priority: mAP50 > mAP50-95 > Precision > Recall
    for k in ["map50", "map50_95", "precision", "recall"]:
        if m1[k] > m2[k] + 1e-6: return 1
        if m2[k] > m1[k] + 1e-6: return 2
    return 0  # tie

def main():
    m1 = Path(MODEL1_PATH)
    m2 = Path(MODEL2_PATH)
    assert m1.exists(), f"Not found: {m1}"
    assert m2.exists(), f"Not found: {m2}"

    # Hashes
    hash1, hash2 = md5(m1), md5(m2)
    print(f"Model1: {m1} | MD5: {hash1}")
    print(f"Model2: {m2} | MD5: {hash2}")

    # Evaluate both
    print("\nEvaluating Model 1...")
    s1 = evaluate(str(m1))
    print("Metrics:", s1)

    print("\nEvaluating Model 2...")
    s2 = evaluate(str(m2))
    print("Metrics:", s2)

    # Pick best
    winner = better(s1, s2)
    os.makedirs(SUBMISSION_FOLDER, exist_ok=True)

    if winner == 1:
        chosen, chosen_stats = m1, s1
        print("\n✅ Model 1 is better")
    elif winner == 2:
        chosen, chosen_stats = m2, s2
        print("\n✅ Model 2 is better")
    else:
        chosen, chosen_stats = m1, s1
        print("\n⚖️ Both models are tied (keeping Model 1 by default)")

    # Copy best.pt into submission_ready
    final_path = Path(SUBMISSION_FOLDER) / "best.pt"
    shutil.copy2(chosen, final_path)
    print(f"Copied best model to {final_path}")

    # Write report
    with open(Path(SUBMISSION_FOLDER) / "comparison_report.txt", "w") as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=======================\n\n")
        f.write(f"Model1: {m1}\nMD5: {hash1}\nMetrics: {s1}\n\n")
        f.write(f"Model2: {m2}\nMD5: {hash2}\nMetrics: {s2}\n\n")
        if winner == 1: f.write("Winner: Model 1\n")
        elif winner == 2: f.write("Winner: Model 2\n")
        else: f.write("Result: Tie (Model 1 kept)\n")

    print(f"Report saved to {SUBMISSION_FOLDER}/comparison_report.txt")

if __name__ == "__main__":
    main()
