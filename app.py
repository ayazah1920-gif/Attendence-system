"""
Online Attendance System using Computer Vision
Backend: Flask + OpenCV (LBPH Face Recognizer)
"""
import os
import cv2
import base64
import numpy as np
import sqlite3
from datetime import datetime, date
from flask import Flask, request, jsonify, render_template, send_file

from database import init_db, get_db
from face_utils import (
    detect_faces,
    train_recognizer,
    load_recognizer,
    recognize_face,
    save_face_samples,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "known_faces", "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "known_faces", "labels.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

init_db()


def decode_base64_image(data_url):
    """Convert a data:image/...;base64,xxxx string into an OpenCV image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/students", methods=["GET"])
def list_students():
    conn = get_db()
    rows = conn.execute("SELECT id, name, roll_no FROM students ORDER BY name").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/register", methods=["POST"])
def register_student():
    """
    Registers a new student by saving multiple face samples captured
    from the browser webcam, then retrains the recognizer.
    Expects JSON: { name, roll_no, images: [dataURL, dataURL, ...] }
    """
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    roll_no = data.get("roll_no", "").strip()
    images = data.get("images", [])

    if not name or not roll_no:
        return jsonify({"success": False, "message": "Name aur Roll No zaroori hain."}), 400
    if len(images) < 3:
        return jsonify({"success": False, "message": "Kam se kam 3 face samples chahiye."}), 400

    conn = get_db()
    cur = conn.execute(
        "INSERT OR IGNORE INTO students (name, roll_no) VALUES (?, ?)", (name, roll_no)
    )
    conn.commit()
    student = conn.execute(
        "SELECT id FROM students WHERE roll_no = ?", (roll_no,)
    ).fetchone()
    student_id = student["id"]
    conn.close()

    saved_count = 0
    student_dir = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)

    for i, data_url in enumerate(images):
        try:
            img = decode_base64_image(data_url)
            faces = detect_faces(img)
            if len(faces) == 0:
                continue
            saved_count += save_face_samples(img, faces, student_dir, prefix=f"{saved_count}")
        except Exception as e:
            print("Sample error:", e)

    if saved_count == 0:
        return jsonify({
            "success": False,
            "message": "Koi face detect nahi hua. Achi lighting mein dobara try karein."
        }), 400

    # Retrain the recognizer with all students' data
    train_recognizer(DATASET_DIR, MODEL_PATH, LABELS_PATH)

    return jsonify({
        "success": True,
        "message": f"{name} register ho gaya ({saved_count} samples).",
        "student_id": student_id
    })


@app.route("/api/recognize", methods=["POST"])
def recognize_and_mark():
    """
    Receives a single webcam frame, recognizes the face, and marks
    today's attendance for that student (once per day).
    Expects JSON: { image: dataURL }
    """
    data = request.get_json(force=True)
    data_url = data.get("image")
    if not data_url:
        return jsonify({"success": False, "message": "Image nahi mili."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"success": False, "message": "Pehle students ko register karein."}), 400

    img = decode_base64_image(data_url)
    faces = detect_faces(img)

    if len(faces) == 0:
        return jsonify({"success": False, "message": "Koi chehra detect nahi hua."})

    recognizer, labels = load_recognizer(MODEL_PATH, LABELS_PATH)
    results = []

    conn = get_db()
    for (x, y, w, h) in faces:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        label_id, confidence = recognize_face(recognizer, face_roi)

        # Lower confidence value = better match in LBPH (it's a distance)
        if label_id is not None and confidence < 70:
            student_id = labels.get(str(label_id))
            if student_id:
                row = conn.execute(
                    "SELECT id, name, roll_no FROM students WHERE id = ?", (student_id,)
                ).fetchone()
                if row:
                    marked = mark_attendance(conn, row["id"])
                    results.append({
                        "name": row["name"],
                        "roll_no": row["roll_no"],
                        "status": "Attendance marked" if marked else "Already marked today",
                        "confidence": round(float(confidence), 1)
                    })
                    continue
        results.append({"name": "Unknown", "status": "Not recognized", "confidence": round(float(confidence), 1)})

    conn.close()
    return jsonify({"success": True, "results": results})


def mark_attendance(conn, student_id):
    today = date.today().isoformat()
    existing = conn.execute(
        "SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id, today)
    ).fetchone()
    if existing:
        return False
    now = datetime.now().strftime("%H:%M:%S")
    conn.execute(
        "INSERT INTO attendance (student_id, date, time) VALUES (?, ?, ?)",
        (student_id, today, now)
    )
    conn.commit()
    return True


@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    day = request.args.get("date", date.today().isoformat())
    conn = get_db()
    rows = conn.execute("""
        SELECT s.name, s.roll_no, a.date, a.time
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (day,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/attendance/export", methods=["GET"])
def export_attendance():
    import csv
    import io
    day = request.args.get("date", date.today().isoformat())
    conn = get_db()
    rows = conn.execute("""
        SELECT s.name, s.roll_no, a.date, a.time
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.date = ?
        ORDER BY s.name
    """, (day,)).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "Roll No", "Date", "Time"])
    for r in rows:
        writer.writerow([r["name"], r["roll_no"], r["date"], r["time"]])
    output.seek(0)

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                      download_name=f"attendance_{day}.csv")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
