from detect_module import detect_faces

if __name__ == "__main__":
    # Nhập đường dẫn ảnh hoặc video để nhận diện
    path = input("Nhập đường dẫn ảnh hoặc video: ")
    
    # Gọi hàm nhận diện đã có
    detect_faces(path=path, model_path="face_data.pkl", display=True)