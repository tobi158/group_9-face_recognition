# collect_faces.py
import cv2
import os


def collect_faces(person_name, save_dir="dataset", max_images=200):
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    person_path = os.path.join(save_dir, person_name)
    os.makedirs(person_path, exist_ok=True)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            file_path = os.path.join(person_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces", frame)
        if cv2.waitKey(1) == ord('q') or count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()

# === Sử dụng ===
# Nhập tên và bắt đầu thu thập
if __name__ == "__main__":
    name = input("Nhập tên người dùng: ")
    collect_faces(name)
