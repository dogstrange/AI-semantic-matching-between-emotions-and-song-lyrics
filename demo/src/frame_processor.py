import numpy as np
from PIL import Image
import io
import base64

EMOTION_LABELS = {"happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"}


def decode_frame(base64_string):
    # strip the prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def analyze_frames(model: object, frames: list[str]) -> list[dict]:
    """
    Accept a batch of RGB numpy frames collected from the web app after a session.
    Returns accumulated emotion scores averaged across all frames that had a detected face,
    in the same format as cam.py: [{"emotion": str, "score": float}, ...]
    """
    score_accumulators = {k: 0.0 for k in EMOTION_LABELS}
    frames_analyzed = 0

    for frame in frames:
        pil_frame = decode_frame(frame)
        rgb = np.array(pil_frame.convert("RGB"))

        try:
            boxes, _ = model.face_detector.detect(pil_frame)
        except Exception:
            continue

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)

            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            try:
                results = model.emotion_classifier(
                    Image.fromarray(face_crop.astype(np.uint8))
                )
                for r in results:
                    emo = r["label"].lower()
                    if emo in score_accumulators:
                        score_accumulators[emo] += r["score"] * 100
                frames_analyzed += 1
            except Exception:
                pass

    if frames_analyzed == 0:
        return []

    final_averages = {
        emo: total / frames_analyzed for emo, total in score_accumulators.items()
    }
    return [
        {"emotion": emo, "score": round(score, 2)}
        for emo, score in final_averages.items()
    ]
