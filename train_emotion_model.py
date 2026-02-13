import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

model = load_model("emotion_model.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Capture quality (does NOT control window size)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

# 🔥 FORCE FULLSCREEN WINDOW
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Emotion Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        preds = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.putText(frame, emotion, (x,y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Emotion Detection", frame)

    # Q or ESC to quit
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()