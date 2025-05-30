from evaluate import evaluate_predictions, plot_roc_auc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import cv2
import face_recognition
import matplotlib.pyplot as plt

# Đường dẫn
EMBEDDING_FILE = "face_data_2.pkl"
TEST_DIR = r"C:\Users\BHXH\Desktop\venv_demo\dataset_split\test"
COSINE_THRESHOLD = 0.6

def load_test_images(test_dir):
    embeddings = []
    labels = []

    if os.path.exists(test_dir):
        for root, _, files in os.walk(test_dir):
            label = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(root, file)
                    image = cv2.imread(filepath)
                    if image is None:
                        print(f"[⚠️] Không đọc được ảnh: {filepath}")
                        continue
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    boxes = face_recognition.face_locations(rgb, model="hog")
                    encodings = face_recognition.face_encodings(rgb, boxes)

                    if len(encodings) == 0:
                        print(f"[⚠️] Không tìm thấy khuôn mặt: {filepath}")
                        continue

                    for enc in encodings:
                        embeddings.append(enc)
                        labels.append(label)
    return embeddings, labels

if __name__ == "__main__":
    # Load dữ liệu đã huấn luyện
    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)
    known_embeddings = data["encodings"]
    known_labels = data["names"]

    # Load ảnh test
    test_embeddings, y_true = load_test_images(TEST_DIR)

    y_pred = []
    y_scores = []

    for emb in test_embeddings:
        sims = cosine_similarity([emb], known_embeddings)[0]
        best_score = np.max(sims)
        best_idx = np.argmax(sims)

        if best_score > COSINE_THRESHOLD:
            pred = known_labels[best_idx]
        else:
            pred = "Unknown"

        y_pred.append(pred)
        y_scores.append(best_score)

    # Tạo nhãn nhị phân cho ROC-AUC
    y_true_binary = [1 if label != "Unknown" else 0 for label in y_true]

    # Đánh giá đa lớp
    evaluate_predictions(
        y_true,
        y_pred,
        labels=list(set(y_true + ["Unknown"])),
        normalize=True
    )

    # Vẽ ROC nếu có đủ nhãn
    if len(set(y_true_binary)) < 2:
        print("⚠️ Không thể vẽ ROC vì thiếu nhãn 'Unknown' hoặc 'Known' trong ground truth.")
    else:
        plot_roc_auc(y_true_binary, y_scores, label_pos="Known")

    # Vẽ histogram phân phối cosine similarity
    known_scores = [s for s, l in zip(y_scores, y_true_binary) if l == 1]
    unknown_scores = [s for s, l in zip(y_scores, y_true_binary) if l == 0]

    if known_scores and unknown_scores:
        plt.hist(known_scores, bins=min(10, len(known_scores)), alpha=0.7, label="Known")
        plt.hist(unknown_scores, bins=min(10, len(unknown_scores)), alpha=0.7, label="Unknown")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Số lượng")
        plt.title("Phân phối cosine similarity")
        plt.legend()
        plt.show()
    else:
        print("⚠️ Không đủ dữ liệu để vẽ histogram.")
