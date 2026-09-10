# 📷 Online Attendance System (Computer Vision)

Face recognition se attendance mark karne wala full-stack system — **Flask backend** + **HTML/JS/CSS frontend**, camera browser mein chalta hai isliye kisi bhi laptop/PC pe demo ho sakta hai.

## Kaise kaam karta hai
- **Detection:** OpenCV Haar Cascade se chehra detect hota hai.
- **Recognition:** OpenCV LBPH (Local Binary Patterns Histograms) Face Recognizer se pehchana jata hai — dlib install karne ka jhanjhat nahi.
- **Storage:** SQLite (`attendance.db`) mein students aur attendance records save hote hain.
- **Frontend:** Browser ka webcam (`getUserMedia`) use hota hai, frame ko base64 image bana ke backend ko bheja jata hai.

## Setup

```bash
cd attendance_system
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Browser mein kholein: **http://127.0.0.1:5000**

## Use karne ka tareeqa

1. **Register New Student** section mein Name aur Roll No likhein.
2. "Capture 5 Samples" button 5 dafa dabayein (har baar thora angle/expression change karein — achi lighting zaroori hai).
3. "Register Student" dabayein — system model ko dobara train kar dega.
4. Attendance mark karne ke liye "Mark Attendance" button dabayein — chehra pehchan kar automatically aaj ki date mein attendance lag jayegi (ek din mein sirf ek dafa).
5. Neeche table mein aaj ki attendance dikhegi, "Export CSV" se download bhi ho sakti hai.

## Project Structure

```
attendance_system/
├── app.py              # Flask backend (routes)
├── database.py         # SQLite schema + connection
├── face_utils.py       # Face detection + LBPH training/recognition
├── requirements.txt
├── templates/
│   └── index.html      # Frontend page
├── static/
│   ├── style.css
│   └── script.js        # Webcam capture + API calls
├── dataset/             # Har student ke face samples (auto-generated)
└── known_faces/         # Trained model (trainer.yml) + labels.json
```

## Notes / Improvements aap kar sakte hain
- Multiple faces ek frame mein bhi handle hoti hain (group photo se bulk attendance).
- Confidence threshold (`< 70`) `app.py` mein adjust kar sakte hain — kam value = strict matching.
- Production ke liye SQLite ki jagah PostgreSQL/MySQL use karein aur HTTPS zaroor lagayein (camera permission ke liye bhi zaroori hai).
- Chahen to `face_recognition` (dlib-based) library pe switch kar ke accuracy behtar kar sakte hain, lekin install thora mushkil hota hai (cmake/dlib build).
