import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from embed import song_collection
from trans_model import ClapEncoder
import requests
import webbrowser
import subprocess
import torch
import numpy as np
import chromadb

clap_collection = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "data", "clap_db")
).get_collection("clap-song-collection")

device = "cuda" if torch.cuda.is_available() else "cpu"

EMO_DIR = os.path.dirname(__file__)
TOOL_DIR = os.path.join(EMO_DIR, "tools")


def rephrase_emotions(emotions: list[str]) -> str:
    prmpt = f"You are describing the emotional journey of a person in short sentence.Given this sorted emotions from high to low in one session {', '.join(emotions)}. I want you to write one short descriptive sentence."
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prmpt,
            "stream": False,
        },  # wait for ollama finish generate before returning
    )
    return response.json()["response"]


def open_song(song, artist):
    search = f"{song} {artist}".replace(" ", "+")
    webbrowser.open(f"https://www.youtube.com/results?search_query={search}")


def play_song(title, artist):
    query = f"{title} {artist}"
    subprocess.run(
        ["mpv", f"ytdl://ytsearch1:{query}"]
    )  # ytdl:// means using ytdl protocol's


def clap_encode_text(emotion_words: list[str]) -> np.ndarray:
    emotion_sen = rephrase_emotions(emotion_words)
    print(f"emotion sentence: \n {emotion_sen}")
    model = ClapEncoder()
    model.model.eval()

    text_inputs = model.process_text(emotion_sen)

    with torch.no_grad():
        text_vector = model.encode_text(text_inputs)
    text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
    return text_vector.cpu().numpy()


def base_id(entry_id: str) -> str:
    return entry_id.removesuffix("_text").removesuffix("_audio")


def clap_query(encoded_emo: np.ndarray):
    """
      song_dict = {
        "john_lennon_imagine":
            {
            "text": 0.82,
            "audio": 0.76,
            "combined": 0.79
            },
        "linkin_park_numb":
            {
            "text": 0.71,
            "combined": 0.71
            },
        "radiohead_creep":
            {
            "audio": 0.68,
            "combined": 0.68
            },
    }
    """

    top_text = clap_collection.query(
        query_embeddings=encoded_emo.tolist(), n_results=10, where={"type": "text"}
    )
    top_audio = clap_collection.query(
        query_embeddings=encoded_emo.tolist(), n_results=10, where={"type": "audio"}
    )
    song_dict = {}

    for i in range(len(top_text["ids"][0])):
        sid = base_id(top_text["ids"][0][i])
        song_dict.setdefault(sid, {})["text"] = top_text["distances"][0][i]

    for i in range(len(top_audio["ids"][0])):
        sid = base_id(top_audio["ids"][0][i])
        song_dict.setdefault(sid, {})["audio"] = top_audio["distances"][0][i]

    for sid, scores in song_dict.items():
        scores["combined"] = sum(scores.values()) / len(scores)

    best = min(song_dict, key=lambda sid: song_dict[sid]["combined"])
    print(song_dict)
    print(f"distance = {song_dict[best]['combined']}")

    return best


def clap_open_song(emotion_arr):
    emo_vec = clap_encode_text(emotion_arr)
    best_song = clap_query(emo_vec)
    play_song(best_song, "")


if __name__ == "__main__":
    # find_and_open_song(["stress", "sad"])
    emo_vec = clap_encode_text(["stress", "sad", "happy"])
    best_song = clap_query(emo_vec)
    play_song(best_song, "")
