import os
import random

def balance_dataset(dataset_path="dataset"):
    folder_counts = {}
    
    # Đếm số ảnh trong mỗi thư mục con
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            folder_counts[folder] = len(image_files)

    # Tìm số lượng ảnh nhỏ nhất
    min_count = min(folder_counts.values())
    print(f"📉 Số ảnh ít nhất trong 1 thư mục: {min_count}\n")

    # Duyệt từng thư mục và xóa bớt ảnh nếu cần
    for folder, count in folder_counts.items():
        folder_path = os.path.join(dataset_path, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if count > min_count:
            to_delete = count - min_count
            files_to_remove = random.sample(image_files, to_delete)
            for file_name in files_to_remove:
                os.remove(os.path.join(folder_path, file_name))
            print(f"🗑️ Đã xóa {to_delete} ảnh trong thư mục '{folder}'")
        else:
            print(f"✅ Thư mục '{folder}' đã đủ {min_count} ảnh")

    print("\n📦 Dataset đã được cân bằng!")

# Gọi hàm
balance_dataset(r"C:\Users\BHXH\Desktop\venv_demo\lfw-funneled\lfw_funneled")  # Đảm bảo đường dẫn đúng với thư mục chứa ảnh của bạn
