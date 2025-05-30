from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
import face_recognition

def evaluate_model(test_dir="dataset_split/test", model_path="face_data.pkl", tolerance=0.6):
    # Load mô hình đã train
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # ✅ Kiểm tra và lọc encoding hợp lệ
        valid_data = [
            (enc, name) for enc, name in zip(data["encodings"], data["names"])
            if isinstance(enc, np.ndarray) and len(enc) == 128
        ]
        if not valid_data:
            print("❌ Không có encoding hợp lệ trong model.")
            return

        known_encodings, known_names = zip(*valid_data)
        known_encodings = list(known_encodings)
        known_names = list(known_names)
        unique_labels = sorted(set(known_names))

    except FileNotFoundError:
        print(f"⚠️ File {model_path} không tồn tại!")
        return


    y_true, y_pred, y_score = [], [], []
    total_images, failed_images = 0, 0

    label_to_index = {name: i for i, name in enumerate(unique_labels)}

    for person_dir in Path(test_dir).iterdir():
        if not person_dir.is_dir():
            continue
        true_name = person_dir.name
        for image_path in person_dir.glob("*"):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            total_images += 1
            try:
                image = face_recognition.load_image_file(image_path)
                locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, locations)
                if not encodings:
                    failed_images += 1
                    continue
                enc = encodings[0]
                if len(enc) != 128:
                    failed_images += 1
                    continue
                distances = face_recognition.face_distance(known_encodings, enc)
                pred_name = "Unknown"
                score = 0
                if distances.size > 0 and distances.min() < tolerance:
                    best_match_index = distances.argmin()
                    pred_name = known_names[best_match_index]
                    score = 1 - distances[best_match_index]  # Sử dụng 1 - distance làm "confidence"
                y_true.append(true_name)
                y_pred.append(pred_name)
                y_score.append(score)
            except Exception as e:
                print(f"Lỗi xử lý {image_path}: {e}")
                failed_images += 1

    print("\n===== 📊 Đánh giá mô hình =====")
    print(f"Tổng ảnh: {total_images}")
    print(f"Ảnh thất bại: {failed_images}")
    print(f"🎯 Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # # === 🧩 Vẽ ma trận nhầm lẫn
    # cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt="d", xticklabels=unique_labels, yticklabels=unique_labels, cmap="Blues")
    # plt.xlabel("Dự đoán")
    # plt.ylabel("Thực tế")
    # plt.title("📉 Ma trận nhầm lẫn")
    # plt.tight_layout()
    # plt.show()

      # === 📈 Vẽ ROC cho nhiều lớp
    if len(unique_labels) > 1:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=unique_labels)
        y_score_bin = np.zeros_like(y_true_bin, dtype=float)

        for i, label in enumerate(unique_labels):
            for j in range(len(y_pred)):
                if y_pred[j] == label:
                    y_score_bin[j][i] = y_score[j]

        # Vẽ đường ROC cho từng lớp
        plt.figure(figsize=(12, 8))
        for i, label in enumerate(unique_labels):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_bin[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("📈 ROC Curve cho từng người")
        plt.legend(loc="lower right", fontsize="small")
        plt.grid()
        plt.tight_layout()
        plt.show()

    else:
        print("🔔 Cần ít nhất 2 lớp để vẽ đường cong ROC.")
    

if __name__ == "__main__":
    evaluate_model("dataset_split/test", "face_data.pkl", tolerance=0.6)
