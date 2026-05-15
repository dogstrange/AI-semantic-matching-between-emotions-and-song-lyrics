import cv2
from PIL import Image
from transformers import pipeline

# Load emotion model
emotion_detector = pipeline(
    "image-classification", model="dima806/facial_emotions_image_detection"
)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        # Crop face
        face = frame[y : y + h, x : x + w]

        # Convert BGR to RGB then to PIL Image
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Run emotion detection
        result = emotion_detector(face_pil)
        label = result[0]["label"]
        score = result[0]["score"]
        print(label, score)
        cv2.putText(frame, f"{label} {score:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw bounding boxes and labels on frame
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
