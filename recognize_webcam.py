# recognize_webcam.py
import cv2
import numpy as np
import torch
import timm
import joblib
from torchvision import transforms

# Load model và nhãn
clf = joblib.load("face_model.pkl")
label_map = joblib.load("label_map.pkl")

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

# Nhận diện webcam
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        emb = get_embedding(face_resized)
        pred = clf.predict([emb])[0]
        prob = np.max(clf.predict_proba([emb]))

        label = label_map[pred]
        text = f"{label} ({prob:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
