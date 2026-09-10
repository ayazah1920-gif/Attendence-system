"""
Online Attendance System using Computer Vision - Streamlit version
"""
import os
import io
import csv
from datetime import date, datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from database import init_db, get_db
from face_utils import (
    detect_faces,
    train_recognizer,
    save_face_samples,
    load_recognizer,
    recognize_face,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "known_faces", "trainer.yml")
LABELS_PATH = os.path.join(BASE_DIR, "known_faces", "labels.json")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
init_db()

st.set_page_config(page_title="Attendance System", page_icon="📷", layout="wide")


def pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


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


st.title("📷 Online Attendance System")
st.caption("Computer Vision based Face Recognition Attendance")

tab1, tab2, tab3 = st.tabs(["✅ Mark Attendance", "🧑‍🎓 Register Student", "📋 Attendance Records"])

# ---------------- Tab 1: Mark Attendance ----------------
with tab1:
    st.subheader("Mark Attendance")
    st.write("Camera se photo lein — system chehra pehchan kar attendance mark kar dega.")

    photo = st.camera_input("Camera", key="attendance_camera")

    if photo is not None:
        if not os.path.exists(MODEL_PATH):
            st.error("Pehle kam az kam ek student register karein.")
        else:
            img = pil_to_cv2(Image.open(photo))
            faces = detect_faces(img)

            if len(faces) == 0:
                st.warning("Koi chehra detect nahi hua. Dobara try karein.")
            else:
                recognizer, labels = load_recognizer(MODEL_PATH, LABELS_PATH)
                conn = get_db()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                for (x, y, w, h) in faces:
                    face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                    label_id, confidence = recognize_face(recognizer, face_roi)

                    if label_id is not None and confidence < 70:
                        student_id = labels.get(str(label_id))
                        row = conn.execute(
                            "SELECT id, name, roll_no FROM students WHERE id = ?", (student_id,)
                        ).fetchone() if student_id else None

                        if row:
                            marked = mark_attendance(conn, row["id"])
                            if marked:
                                st.success(f"✅ {row['name']} ({row['roll_no']}) — Attendance marked!")
                            else:
                                st.info(f"ℹ️ {row['name']} ({row['roll_no']}) — Already marked today.")
                            continue

                    st.error("❌ Chehra pehchana nahi gaya (Unknown).")
                conn.close()

# ---------------- Tab 2: Register Student ----------------
with tab2:
    st.subheader("Register New Student")
    st.write("Naam aur Roll No likhein, phir kam az kam 3 (behtar 5) alag-alag angles se photo capture karein.")

    name = st.text_input("Full Name")
    roll_no = st.text_input("Roll No")

    if "samples" not in st.session_state:
        st.session_state.samples = []

    st.write(f"**Samples captured: {len(st.session_state.samples)} / 5**")

    reg_photo = st.camera_input("Capture a face sample", key=f"reg_camera_{len(st.session_state.samples)}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add this sample") and reg_photo is not None:
            st.session_state.samples.append(reg_photo)
            st.rerun()
    with col2:
        if st.button("🗑️ Clear samples"):
            st.session_state.samples = []
            st.rerun()

    if st.button("Register Student", type="primary", disabled=len(st.session_state.samples) < 3):
        if not name.strip() or not roll_no.strip():
            st.error("Name aur Roll No zaroori hain.")
        else:
            conn = get_db()
            conn.execute(
                "INSERT OR IGNORE INTO students (name, roll_no) VALUES (?, ?)",
                (name.strip(), roll_no.strip())
            )
            conn.commit()
            student = conn.execute(
                "SELECT id FROM students WHERE roll_no = ?", (roll_no.strip(),)
            ).fetchone()
            student_id = student["id"]
            conn.close()

            student_dir = os.path.join(DATASET_DIR, str(student_id))
            os.makedirs(student_dir, exist_ok=True)

            saved = 0
            for i, photo in enumerate(st.session_state.samples):
                img = pil_to_cv2(Image.open(photo))
                faces = detect_faces(img)
                if len(faces) == 0:
                    continue
                saved += save_face_samples(img, faces, student_dir, prefix=f"{saved}")

            if saved == 0:
                st.error("Koi face detect nahi hua samples mein. Achi lighting mein dobara try karein.")
            else:
                train_recognizer(DATASET_DIR, MODEL_PATH, LABELS_PATH)
                st.success(f"{name} register ho gaya ({saved} samples se train hua)!")
                st.session_state.samples = []
                st.rerun()

# ---------------- Tab 3: Attendance Records ----------------
with tab3:
    st.subheader("Attendance Records")
    selected_date = st.date_input("Date", value=date.today())

    conn = get_db()
    rows = conn.execute("""
        SELECT s.name, s.roll_no, a.date, a.time
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (selected_date.isoformat(),)).fetchall()
    conn.close()

    if rows:
        import pandas as pd
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, use_container_width=True)

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Name", "Roll No", "Date", "Time"])
        for r in rows:
            writer.writerow([r["name"], r["roll_no"], r["date"], r["time"]])

        st.download_button(
            "⬇️ Export CSV",
            data=csv_buffer.getvalue(),
            file_name=f"attendance_{selected_date.isoformat()}.csv",
            mime="text/csv"
        )
    else:
        st.info("Is date ke liye koi attendance record nahi mila.")

    st.divider()
    st.subheader("All Registered Students")
    conn = get_db()
    students = conn.execute("SELECT name, roll_no FROM students ORDER BY name").fetchall()
    conn.close()
    if students:
        import pandas as pd
        st.dataframe(pd.DataFrame([dict(s) for s in students]), use_container_width=True)
    else:
        st.info("Abhi tak koi student register nahi hua.")