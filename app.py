# --- app.py ---

import flask
from flask import Flask, jsonify, request, render_template, send_from_directory
import platform
import os
import logging
import base64
from io import BytesIO
from PIL import Image
import face_recognition
import cv2
import numpy as np
import pickle
import csv
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# --- Config ---
# We will save encodings in a .pkl file for efficiency
ENCODINGS_FILE = "data/encodings.pkl" 
ATTENDANCE_DIR = "data/attendance"
KNOWN_DIR = "data/known_faces"

# Create directories if they don't exist
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attendance_app")

# --- Helper Functions (From your utils) ---

def save_encodings(encodings, names, filepath):
    """Saves face encodings and names to a pickle file."""
    data = {"encodings": encodings, "names": names}
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Encodings saved to {filepath}")

def load_encodings(filepath):
    """Loads face encodings and names from a pickle file."""
    if not os.path.exists(filepath):
        return [], []
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        # Ensure keys exist, default to empty lists
        return data.get("encodings", []), data.get("names", [])
    except Exception as e:
        logger.error(f"Error loading encodings: {e}")
        return [], []

def encode_known_faces(known_faces_dir):
    """
    Encodes all faces in the known_faces directory.
    Structure: known_faces/
                /Alice/
                    1.jpg
                    2.jpg
                /Bob/
                    1.jpg
    """
    known_encodings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        return known_encodings, known_names
        
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        logger.info(f"Processing images for {person_name}...")
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Find face locations (get the first face)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    # Get the encoding for the first face found
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                else:
                    logger.warning(f"No face found in {image_path}, skipping.")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                
    return known_encodings, known_names

def mark_attendance(names, directory):
    """
    Marks attendance in a CSV file for the current day.
    Creates a new file for each day.
    Avoids duplicate entries for the same name on the same day.
    """
    # 1. Filename uses today's date
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"
    filepath = os.path.join(directory, filename)
    
    # 2. Timestamp is the exact time of recognition
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Read existing names from today's file to avoid duplicates
    existing_names = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                for row in reader:
                    if len(row) > 1:
                        existing_names.add(row[1]) # Add name from "Name" column
        except Exception as e:
            logger.error(f"Could not read existing attendance file: {e}")

    # Open in 'a' (append) mode to add new entries
    new_entries_added = False
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new/empty
        if not existing_names and os.path.getsize(filepath) == 0:
            writer.writerow(["Timestamp", "Name"])
        
        # Write new names
        for name in names:
            if name != "Unknown" and name not in existing_names:
                writer.writerow([timestamp, name])
                existing_names.add(name) # Add to set to avoid duplicates in this same run
                new_entries_added = True
                
    return filename, new_entries_added # Return the name of the file

# --- Load Encodings on Startup ---
known_encodings, known_names = load_encodings(ENCODINGS_FILE)
if not known_encodings or not known_names:
    logger.info("No encodings file found or file is empty. Encoding faces from directory...")
    known_encodings, known_names = encode_known_faces(KNOWN_DIR)
    if known_encodings: # Only save if we found faces
        save_encodings(known_encodings, known_names, ENCODINGS_FILE)
else:
    logger.info(f"Loaded {len(known_names)} known faces from {ENCODINGS_FILE}.")


# --- Flask Routes ---

@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "Running",
        "python": platform.python_version(),
        "faces_loaded": len(known_names)
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Receives a Base64-encoded image from the frontend,
    recognizes faces, and marks attendance.
    """
    global known_encodings, known_names
    
    data = request.get_json()
    image_data = data.get('image') # Expecting JSON: {"image": "base64_string..."}

    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided'}), 400
    
    recognized_names = set()

    try:
        # Decode the Base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        img_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Convert PIL Image to numpy array (RGB)
        rgb_frame = np.array(img_pil) 

        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for enc in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            name = "Unknown"
            
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                recognized_names.add(name)

        if recognized_names:
            file, added = mark_attendance(recognized_names, ATTENDANCE_DIR)
            message = f"Attendance marked for: {', '.join(recognized_names)}."
            if not added:
                message = f"{', '.join(recognized_names)} already marked present today."
            
            return jsonify({
                "recognized": list(recognized_names), 
                "attendance_file": file,
                "message": message
            })
        
        return jsonify({"recognized": [], "message": "No known faces recognized."})

    except Exception as e:
        logger.error(f"Error in /recognize: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/register', methods=['POST'])
def register_face():
    """
    Registers a new face. Receives name and Base64-encoded image.
    Saves the image and re-trains the model.
    """
    global known_encodings, known_names
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')

    if not name or not image_data:
        return jsonify({'status': 'error', 'message': 'Name or image missing'}), 400
    
    # Sanitize name to prevent directory traversal
    name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not name:
        return jsonify({'status': 'error', 'message': 'Invalid name provided.'}), 400

    try:
        # Decode image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Convert to numpy array for face detection
        img_np = np.array(img)
        
        # Check if at least one face is present
        face_locations = face_recognition.face_locations(img_np)
        if not face_locations:
            return jsonify({'status': 'error', 'message': 'No face detected in the image.'}), 400

        # Save image to data/known_faces/<name>/
        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        file_count = len(os.listdir(person_dir)) + 1
        file_path = os.path.join(person_dir, f'{file_count}.jpg')
        img.save(file_path, "JPEG")

        # Re-encode all faces and save
        logger.info("New face registered. Re-encoding all known faces...")
        known_encodings, known_names = encode_known_faces(KNOWN_DIR)
        save_encodings(known_encodings, known_names, ENCODINGS_FILE)
        logger.info("Re-encoding complete.")

        return jsonify({'status': 'success', 'message': f'Face registered for {name}!'})
    except Exception as e:
        logger.error(f"Error in /register: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    """
    Serves the attendance CSV file for download.
    """
    return send_from_directory(
        ATTENDANCE_DIR,
        filename,
        as_attachment=True 
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)