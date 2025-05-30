import os
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# Thiết lập mô hình CNN
model = timm.create_model("resnest50d", pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Hàm tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Hàm trích xuất embedding từ ảnh
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy().flatten()
    embedding /= np.linalg.norm(embedding)
    return embedding

# Đường dẫn dữ liệu
dataset_dir = r"C:\Users\BHXH\Desktop\venv_demo\lfw-funneled\lfw_funneled"
X = []
y = []

# Duyệt qua các thư mục con (mỗi thư mục là một người)
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_path):
        for file_name in os.listdir(person_path):
            if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                file_path = os.path.join(person_path, file_name)
                try:
                    emb = extract_embedding(file_path)
                    X.append(emb)
                    y.append(person_name)
                except Exception as e:
                    print(f"Lỗi xử lý {file_path}: {e}")

# Mã hóa nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train SVM
print("🔍 Đang huấn luyện mô hình SVM...")
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y_encoded)

# Lưu mô hình và nhãn
joblib.dump(clf, "svm_model.pkl")
label_map = {idx: label for idx, label in enumerate(le.classes_)}
joblib.dump(label_map, "label_map.pkl")

print("✅ Đã huấn luyện xong và lưu vào svm_model.pkl và label_map.pkl")
print("📦 Số lớp:", len(label_map))
