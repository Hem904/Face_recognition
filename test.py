import cv2
import face_recognition

Kathan = cv2.imread("known_faces/Kathan.jpg")
rgb_img = cv2.cvtColor(Kathan, cv2.COLOR_BGR2RGB)
Kathan_encoding = face_recognition.face_encodings(rgb_img)[0]

Naisargi = cv2.imread("known_faces/Naisargi.jpg")
rgb_img2 = cv2.cvtColor(Naisargi, cv2.COLOR_BGR2RGB)
Naisargi_encoding = face_recognition.face_encodings(rgb_img2)[0]

Archi = cv2.imread("known_faces/Archi.jpg")
rgb_img3 = cv2.cvtColor(Archi, cv2.COLOR_BGR2RGB)
Archi_encoding = face_recognition.face_encodings(rgb_img3)[0]

Akshat = cv2.imread("known_faces/Akshat.jpg")
rgb_img4 = cv2.cvtColor(Akshat, cv2.COLOR_BGR2RGB)
Akshat_encoding = face_recognition.face_encodings(rgb_img4)[0]

Jay = cv2.imread("known_faces/Jay.jpg")
rgb_img5 = cv2.cvtColor(Jay, cv2.COLOR_BGR2RGB)
Jay_encoding = face_recognition.face_encodings(rgb_img5)[0]

Kriti = cv2.imread("known_faces/Kriti.jpg")
rgb_img6 = cv2.cvtColor(Kriti, cv2.COLOR_BGR2RGB)
Kriti_encoding = face_recognition.face_encodings(rgb_img6)[0]

Jaymin = cv2.imread("known_faces/Jaymin.jpg")
rgb_img7 = cv2.cvtColor(Jaymin, cv2.COLOR_BGR2RGB)
Jaymin_encoding = face_recognition.face_encodings(rgb_img7)[0]

Kashish = cv2.imread("known_faces/Kashish.jpg")
rgb_img8 = cv2.cvtColor(Kashish, cv2.COLOR_BGR2RGB)
Kashish_encoding = face_recognition.face_encodings(rgb_img8)[0]

Tiwari = cv2.imread("known_faces/Tiwari.jpg")
rgb_img9 = cv2.cvtColor(Tiwari, cv2.COLOR_BGR2RGB)
Tiwari_encoding = face_recognition.face_encodings(rgb_img9)[0]

result = face_recognition.compare_faces([Kathan_encoding], Kathan_encoding)
print(result)

cv2.imshow("Kathan", Kathan)
cv2.imshow("Naisargi", Naisargi)
cv2.imshow("Archi", Archi)
cv2.imshow("Akshat", Akshat)
cv2.imshow("Jay", Jay)
cv2.imshow("Kriti", Kriti)
cv2.imshow("Jaymin", Jaymin)
cv2.imshow("Kashish", Kashish)
cv2.imshow("Tiwari", Tiwari)

cv2.waitKey(0)