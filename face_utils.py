"""
Face detection (Haar Cascade) + recognition (LBPH) utilities.
No dlib dependency needed - runs with plain opencv-contrib-python.
"""
import os
import json
import cv2
import numpy as np

CASCADE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades", "haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

if FACE_CASCADE.empty():
    raise RuntimeError(f"Haar cascade file load nahi hui: {CASCADE_PATH}")


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    return faces


def save_face_samples(img, faces, out_dir, prefix="0"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    saved = 0
    for i, (x, y, w, h) in enumerate(faces):
        face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        path = os.path.join(out_dir, f"{prefix}_{i}.jpg")
        cv2.imwrite(path, face)
        saved += 1
    return saved


def train_recognizer(dataset_dir, model_path, labels_path):
    """
    dataset_dir/
        <student_id>/
            0_0.jpg, 1_0.jpg, ...
    Trains an LBPH recognizer over all students' samples.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    ids = []
    labels = {}  # numeric label -> student_id (string)

    student_dirs = sorted(os.listdir(dataset_dir))
    for label_num, student_id in enumerate(student_dirs):
        student_path = os.path.join(dataset_dir, student_id)
        if not os.path.isdir(student_path):
            continue
        labels[str(label_num)] = student_id
        for filename in os.listdir(student_path):
            if not filename.lower().endswith((".jpg", ".png")):
                continue
            img_path = os.path.join(student_path, filename)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            face_samples.append(gray)
            ids.append(label_num)

    if not face_samples:
        return False

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(model_path)
    with open(labels_path, "w") as f:
        json.dump(labels, f)
    return True


def load_recognizer(model_path, labels_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(labels_path, "r") as f:
        labels = json.load(f)
    return recognizer, labels


def recognize_face(recognizer, face_gray_200x200):
    label_id, confidence = recognizer.predict(face_gray_200x200)
    return label_id, confidence