import face_recognition
import os
import pickle

def train_model():
    dataset_path = "dataset"
    known_encodings = []
    known_names = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)

    data = {"encodings": known_encodings, "names": known_names}

    os.makedirs("models", exist_ok=True)

    with open("models/encodings.pickle", "wb") as f:
        pickle.dump(data, f)

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
