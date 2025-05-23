# app.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import torch
import timm
import cv2
from PIL import Image
from io import BytesIO
from torchvision import transforms
import os
import shutil

app = FastAPI()

# Load model
model = timm.create_model("resnest50d", pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Load classifier and label map (if exists)
clf = joblib.load("face_model.pkl")
label_map = joblib.load("label_map.pkl")

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

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        emb = get_embedding(img_np)
        probs = clf.predict_proba([emb])[0]
        pred = np.argmax(probs)
        label = label_map[pred]
        confidence = float(probs[pred])

        return {"result": label, "confidence": round(confidence, 2)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/users/")
def list_users():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        return []
    return [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

@app.post("/register/")
async def register_user(name: str = Form(...), file: UploadFile = File(...)):
    user_dir = os.path.join("dataset", name)
    os.makedirs(user_dir, exist_ok=True)
    img_id = len(os.listdir(user_dir))
    img_path = os.path.join(user_dir, f"{img_id}.jpg")
    contents = await file.read()
    with open(img_path, 'wb') as f:
        f.write(contents)
    return {"message": f"Đã lưu ảnh cho người dùng '{name}'"}

@app.delete("/users/{name}/")
def delete_user(name: str):
    user_dir = os.path.join("dataset", name)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        return {"message": f"Đã xoá người dùng '{name}'"}
    return JSONResponse(status_code=404, content={"error": "Người dùng không tồn tại"})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
