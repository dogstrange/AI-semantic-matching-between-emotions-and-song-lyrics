import numpy as np
from PIL import Image
import io
import base64

EMOTION_LABELS = {"happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"}

# Weighted blends of the 7 base emotions into a richer vocabulary.
# Per frame: complex_score = sum(weight * prob[base]) for each base in the blend.
COMPLEX_EMOTIONS = {
    "joyful":      {"happy": 1.0},
    "excited":     {"happy": 0.6, "surprise": 0.4},
    "euphoric":    {"happy": 0.7, "surprise": 0.3},
    "content":     {"happy": 0.5, "neutral": 0.5},
    "hopeful":     {"happy": 0.6, "neutral": 0.2, "surprise": 0.2},
    "serene":      {"neutral": 0.6, "happy": 0.4},
    "peaceful":    {"neutral": 1.0},
    "calm":        {"neutral": 0.7, "happy": 0.3},
    "dreamy":      {"neutral": 0.4, "happy": 0.3, "surprise": 0.3},
    "bittersweet": {"happy": 0.5, "sad": 0.5},
    "nostalgic":   {"sad": 0.4, "happy": 0.4, "neutral": 0.2},
    "wistful":     {"sad": 0.5, "happy": 0.3, "neutral": 0.2},
    "melancholic": {"sad": 0.7, "neutral": 0.3},
    "lonely":      {"sad": 0.5, "neutral": 0.5},
    "gloomy":      {"sad": 0.6, "disgust": 0.2, "neutral": 0.2},
    "heartbroken": {"sad": 0.8, "fear": 0.2},
    "vulnerable":  {"fear": 0.5, "sad": 0.5},
    "anxious":     {"fear": 0.7, "surprise": 0.3},
    "tense":       {"fear": 0.5, "angry": 0.5},
    "restless":    {"surprise": 0.4, "fear": 0.3, "angry": 0.3},
    "frustrated":  {"angry": 0.6, "disgust": 0.4},
    "irritated":   {"angry": 0.5, "disgust": 0.5},
    "bitter":      {"disgust": 0.5, "angry": 0.3, "sad": 0.2},
    "contempt":    {"disgust": 0.6, "angry": 0.4},
    "awestruck":   {"surprise": 0.7, "happy": 0.3},
}


def map_to_complex(base_probs: dict) -> dict:
    """Map a 7-emotion probability dict to a complex-emotion score dict."""
    return {
        name: sum(w * base_probs.get(base, 0.0) for base, w in blend.items())
        for name, blend in COMPLEX_EMOTIONS.items()
    }


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
    score_accumulators = {k: 0.0 for k in COMPLEX_EMOTIONS}
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
                base_probs = {
                    r["label"].lower(): r["score"]
                    for r in results
                    if r["label"].lower() in EMOTION_LABELS
                }
                complex_scores = map_to_complex(base_probs)
                for name, score in complex_scores.items():
                    score_accumulators[name] += score * 100
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
