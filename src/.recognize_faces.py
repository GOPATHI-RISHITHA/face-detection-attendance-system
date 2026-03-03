import face_recognition
import cv2
import pickle
import pandas as pd
from datetime import datetime
import os

def recognize_faces():
    with open("models/encodings.pickle", "rb") as f:
        data = pickle.load(f)

    cap = cv2.VideoCapture(0)

    os.makedirs("attendance", exist_ok=True)
    attendance_file = "attendance/attendance.csv"

    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(attendance_file, index=False)

    marked_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            matches = face_recognition.compare_faces(
                data["encodings"], encoding
            )

            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = data["names"][matched_idx]

                if name not in marked_names:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.read_csv(attendance_file)
                    df.loc[len(df)] = [name, now]
                    df.to_csv(attendance_file, index=False)
                    marked_names.add(name)

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
