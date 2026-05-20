from cam import run_emotion_session
from emotions import find_and_open_song, clap_open_song

CIRCUMPLEX_WORDS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "stressed",
    "fear": "nervous",
    "disgust": "upset",
    "surprise": "excited",
    "neutral": "calm",
}

def top_emotions(session_results: list[dict], top_n: int = 3) -> list[str]:
    sorted_emotions = sorted(session_results, key=lambda x: x["score"], reverse=True)
    return [
        CIRCUMPLEX_WORDS[e["emotion"]]
        for e in sorted_emotions[:top_n]
        if e["emotion"] in CIRCUMPLEX_WORDS
    ]  # List comprehension


if __name__ == "__main__":
    session_results = run_emotion_session(10)

    if not session_results:
        print("No emotion data captured. Exiting.")
    else:
        emotion_words = top_emotions(session_results, top_n=4)
        print(f"Top emotions: {emotion_words}")
        clap_open_song(emotion_words)
