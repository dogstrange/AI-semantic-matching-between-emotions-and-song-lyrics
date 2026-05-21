import cv2
import time
from facenet_pytorch import MTCNN
from transformers import pipeline
from PIL import Image

EMOTION_LABELS = {"happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"}

face_detector = MTCNN(keep_all=True, device="cpu", post_process=False)
emotion_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
)


def run_emotion_session(duration=30):
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    score_accumulators = {k: 0.0 for k in EMOTION_LABELS}
    frames_analyzed = 0
    session_results_list = []

    print(f"Session Started: Analyzing for {duration} seconds...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)
            boxes, _ = face_detector.detect(pil_frame)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    face_crop = rgb[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    try:
                        results = emotion_classifier(Image.fromarray(face_crop))
                        dominant = results[0]["label"].lower()

                        for r in results:
                            emo = r["label"].lower()
                            if emo in score_accumulators:
                                score_accumulators[emo] += r["score"] * 100

                        frames_analyzed += 1

                        cv2.putText(
                            frame,
                            dominant,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                    except Exception:
                        pass
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"Time: {int(duration - elapsed_time)}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Emotion Analysis Session", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # flush the OpenCV event loop so the window actually closes on macOS
        for _ in range(5):
            cv2.waitKey(1)

    if frames_analyzed > 0:
        final_averages = {
            emo: total / frames_analyzed for emo, total in score_accumulators.items()
        }
        for emo, score in final_averages.items():
            session_results_list.append({"emotion": emo, "score": round(score, 2)})

        print("\n Webcam session Complete. Data stored for export.")
        return session_results_list
    else:
        print("\nNo data collected.")
        return []


if __name__ == "__main__":
    my_emotion_data = run_emotion_session(30)
    print("Results passed to next module:", my_emotion_data)
