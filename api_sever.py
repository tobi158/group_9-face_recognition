from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from fastapi import Query
import os
from detect_module import build_face_dataset, detect_faces  # dùng từ mã gốc

app = FastAPI()

DATASET_FOLDER = r"C:\Users\BHXH\Desktop\venv_demo\lfw-funneled\lfw_funneled"
ENCODING_FILE = "face_data.pkl"
OUTPUT_FOLDER = "output/faces"

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_faces_in_folder(folder_path, model_path="face_data.pkl", display=False):
    results = []

    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue  # bỏ qua file, chỉ xử lý thư mục con

        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(subfolder_path, filename)
                try:
                    # Gọi hàm nhận diện khuôn mặt
                    detect_faces(path=file_path, model_path=model_path, display=display)

                    # Đọc metadata sau khi xử lý
                    with open("output/metadata.json", "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    results.append({
                        "class": subfolder_name,
                        "file": filename,
                        "detected_faces": metadata.get("total", 0),
                        "faces": metadata.get("faces", [])
                    })

                except Exception as e:
                    results.append({
                        "class": subfolder_name,
                        "file": filename,
                        "error": str(e)
                    })

    return results


# API khởi tạo dữ liệu khuôn mặt
@app.post("/build_dataset/")
def api_build_dataset():
    dataset_folder = r"C:\Users\BHXH\Desktop\venv_demo\lfw-funneled\lfw_funneled"
    build_face_dataset(dataset_folder=dataset_folder, output_file="face_data.pkl")
    return {"message": "Dữ liệu khuôn mặt đã được tạo."}

# API nhận diện khuôn mặt từ ảnh
@app.post("/detect_face/")
async def api_detect_face(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    detect_faces(path=temp_file_path, model_path="face_data.pkl", display=False)

    os.remove(temp_file_path)  # xoá file tạm sau khi xử lý

    with open("output/metadata.json", "r", encoding="utf-8") as f:
        metadata = f.read()
    return JSONResponse(content=metadata)

#API show all ảnh đã nhận diện
@app.get("/api/show_faces")
def show_detected_faces():
    folder = "output/faces"
    if not os.path.exists(folder):
        return JSONResponse(content={"message": "Thư mục output/faces không tồn tại."}, status_code=404)

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', 'jpeg'))]
    files.sort()
    return {"faces": files, "total": len(files)}

#API thêm ảnh vào tập huấn luyện
@app.post("/api/update_dataset")
def update_dataset(file: UploadFile = File(...)):
    filename = file.filename
    file_path = os.path.join(DATASET_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": f"Đã cập nhật tập huấn luyện với {filename}"}

#API để xóa file trong tập output/faces
@app.delete("/api/delete_face/")
def delete_face():
    delete = False
    for filename in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            delete = True
    if delete:
        return {"message": f"Đã xóa lịch sử nhận diện"}
    return JSONResponse(status_code=404, content={"error": "không có file nào đã nhận diện."})
from fastapi import Query

@app.post("/api/detect_faces_from_folder/")
def detect_faces_from_folder(folder: str = Query(..., description="Đường dẫn thư mục chứa ảnh test")):
    if not os.path.exists(folder):
        return JSONResponse(status_code=404, content={"error": "Thư mục không tồn tại."})
    
    result = detect_faces_in_folder(folder_path=folder, model_path=ENCODING_FILE)
    return JSONResponse(content=result)
