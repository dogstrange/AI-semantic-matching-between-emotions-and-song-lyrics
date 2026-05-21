from cam import run_emotion_session
from emotions import clap_open_song, top_emotions


if __name__ == "__main__":
    session_results = run_emotion_session(10)

    if not session_results:
        print("No emotion data captured. Exiting.")
    else:
        emotion_words = top_emotions(session_results, top_n=4)
        print(f"Top emotions: {emotion_words}")
        clap_open_song(emotion_words)
