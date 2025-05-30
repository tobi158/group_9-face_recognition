import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
import joblib
import face_recognition

# Load mô hình CNN (ResNeSt-50)
model = timm.create_model("resnest50d", pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Load SVM và label map
clf = joblib.load("svm_model.pkl")
label_map = joblib.load("label_map.pkl")
inv_label_map = {v: k for k, v in label_map.items()}

# Hàm chuyển ảnh về embedding
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(face_img):
    with torch.no_grad():
        img_tensor = transform(face_img).unsqueeze(0)
        emb = model(img_tensor).squeeze().numpy().flatten()
        emb /= np.linalg.norm(emb)
        return emb

# Mở webcam
cap = cv2.VideoCapture(0)

print("🚀 Đang chạy webcam nhận diện khuôn mặt... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # giảm kích thước để tăng tốc
    rgb_small_frame = small_frame[:, :, ::-1]  # BGR -> RGB

    # Tìm khuôn mặt
    face_locations = face_recognition.face_locations(rgb_small_frame)

    for top, right, bottom, left in face_locations:
        # Scale lại tọa độ vì frame đang resize 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        face_img = frame[top:bottom, left:right]
        if face_img.size == 0:
            continue

        # Trích xuất embedding
        try:
            embedding = get_embedding(face_img)
        except:
            continue

        # Dự đoán
        probs = clf.predict_proba([embedding])[0]
        best_class = np.argmax(probs)
        confidence = probs[best_class]
        name = label_map[best_class] if confidence > 0.5 else "Unknown"

        # Hiển thị
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("🔍 Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
