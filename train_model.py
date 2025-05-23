# train_model.py
import os
import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Load ResNeSt-50
model = timm.create_model("resnest50d", pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(img):
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0)
        emb = model(img_tensor).squeeze().numpy().flatten()
        emb /= np.linalg.norm(emb)
        return emb

def load_dataset(dataset_path="dataset"):
    X, y = [], []
    label_map = {}
    label_id = 0
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        label_map[label_id] = person
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = get_embedding(img)
            X.append(emb)
            y.append(label_id)
        label_id += 1
    return np.array(X), np.array(y), label_map

# Huấn luyện
X, y, label_map = load_dataset()

if len(X) == 0 or len(np.unique(y)) < 2:
    print("❌ Không đủ dữ liệu để huấn luyện. Hãy chắc chắn rằng dataset/ có ảnh của ít nhất 2 người.")
    exit()

print(f"✅ Dữ liệu huấn luyện: {len(X)} ảnh từ {len(np.unique(y))} người.")

clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)


# Dự đoán để đánh giá
y_pred = clf.predict(X)

# Báo cáo phân loại
print("\n📊 Báo cáo phân loại:")
print(classification_report(y, y_pred, target_names=[label_map[i] for i in sorted(label_map.keys())]))

# Tính F1 Score (macro)
f1 = f1_score(y, y_pred, average='macro')
print(f"🎯 F1 Score (macro): {f1:.4f}")

# # Vẽ ma trận nhầm lẫn
# cm = confusion_matrix(y, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_map[i] for i in sorted(label_map.keys())])
# disp.plot(cmap='Blues')
# plt.title("🔍 Confusion Matrix")
# plt.tight_layout()
# plt.show()


# Binarize nhãn để dùng ROC (one-vs-rest)
classes = sorted(label_map.keys())
y_bin = label_binarize(y, classes=classes)
y_score = clf.predict_proba(X)
n_classes = y_bin.shape[1]

# Tính ROC và AUC từng lớp
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Vẽ biểu đồ ROC từng lớp
colors = cycle(['blue', 'green', 'red', 'orange', 'purple', 'brown'])
plt.figure()
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{label_map[i]} (AUC = {roc_auc[i]:0.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("🎯 ROC Curve - One-vs-Rest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Lưu model và nhãn
joblib.dump(clf, "face_model.pkl")
joblib.dump(label_map, "label_map.pkl")

print("✅ Đã huấn luyện và lưu mô hình.")
