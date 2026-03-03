import cv2
import os

def capture_faces():
    name = input("Enter Student Name: ").strip()

    dataset_path = os.path.join("dataset", name)
    os.makedirs(dataset_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_path}/{count}.jpg", face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces - Press Q to Quit", frame)

        if count >= 50:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured {count} images successfully!")

if __name__ == "__main__":
    capture_faces()
