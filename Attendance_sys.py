import cv2
import face_recognition
import os

# Load and encode known faces
known_faces = []
known_names = []

face_dir = "known_faces"

for filename in os.listdir(face_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        path = os.path.join(face_dir, filename)

        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(name)
        else:
            print(f"[WARNING] No face found in {filename}")

print(f"[INFO] Loaded encodings for: {known_names}")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect and encode faces in frame
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    # Compare with known faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if matches:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
