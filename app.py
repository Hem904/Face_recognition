from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

app = Flask(__name__)

KNOWN_DIR = "known_faces"
ENCODING_FILE = "encodings/encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"

if not os.path.exists("encodings"):
    os.makedirs("encodings")

known_encodings = []
known_names = []

# ---------------------- LOAD ENCODINGS ----------------------
def load_encodings():
    global known_encodings, known_names
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
            known_encodings = data["encodings"]
            known_names = data["names"]

# ---------------------- SAVE ENCODINGS ----------------------
def save_encodings():
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

load_encodings()

# ---------------------- BASIC API 1 ----------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"}), 200

# ---------------------- BASIC API 2 ----------------------
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")
    if not name or not file:
        return jsonify({"error": "Name and image required"}), 400

    filepath = os.path.join(KNOWN_DIR, f"{name}.jpg")
    file.save(filepath)

    img = cv2.imread(filepath)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return jsonify({"error": "No face detected in image"}), 400

    known_encodings.append(encodings[0])
    known_names.append(name)
    save_encodings()

    return jsonify({"message": f"User {name} registered successfully"}), 200

# ---------------------- BASIC API 3 ----------------------
@app.route("/known-faces", methods=["GET"])
def get_known_faces():
    return jsonify({"users": list(set(known_names))})

# ---------------------- ADVANCED API 1 ----------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    video = cv2.VideoCapture(0)
    recognized = set()

    for _ in range(5):  # Read few frames for accuracy
        ret, frame = video.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for enc in face_encs:
            matches = face_recognition.compare_faces(known_encodings, enc)
            if True in matches:
                idx = np.argmin(face_recognition.face_distance(known_encodings, enc))
                recognized.add(known_names[idx])
    
    video.release()
    return jsonify({"recognized": list(recognized)}), 200

# ---------------------- ADVANCED API 2 ----------------------
@app.route("/attendance", methods=["POST"])
def mark_attendance():
    video = cv2.VideoCapture(0)
    marked = set()

    for _ in range(5):
        ret, frame = video.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encs = face_recognition.face_encodings(rgb, face_locations)

        for enc in face_encs:
            matches = face_recognition.compare_faces(known_encodings, enc)
            if True in matches:
                idx = np.argmin(face_recognition.face_distance(known_encodings, enc))
                name = known_names[idx]
                if name not in marked:
                    marked.add(name)
                    with open(ATTENDANCE_FILE, "a") as f:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{name},{now}\n")

    video.release()
    return jsonify({"attendance_marked": list(marked)}), 200

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app.run(debug=True)
