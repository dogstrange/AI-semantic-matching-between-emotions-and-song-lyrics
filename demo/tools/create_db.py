import os
import re

import chromadb
from datasets import load_dataset

from encode_clap import encode_clap

TOOLS_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(TOOLS_DIR, "..", "data", "audio")
META_DIR = os.path.join(TOOLS_DIR, "..", "data", "metadata")
CLAP_DB_PATH = os.path.join(TOOLS_DIR, "..", "data", "clap_db")

client = chromadb.PersistentClient(path=CLAP_DB_PATH)
clap_collection = client.get_or_create_collection(
    name="clap-song-collection", metadata={"hnsw:space": "cosine"}
)


def safe_id(title: str, artist: str) -> str:
    raw = f"{artist}_{title}".lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def add_song(sample):
    title = sample["song"]
    artist = sample["artist"]
    lyrics = sample["text"]

    song_id = safe_id(title, artist)
    audio_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")

    if not os.path.exists(audio_path):
        return sample

    existing = clap_collection.get(ids=[f"{song_id}_audio"])
    if existing["ids"]:
        return sample

    text_vector, audio_vector = encode_clap(lyrics, audio_path)

    clap_collection.add(
        ids=[f"{song_id}_audio", f"{song_id}_text"],
        embeddings=[audio_vector.tolist(), text_vector.tolist()],
        metadatas=[
            {"title": title, "artist": artist, "type": "audio"},
            {"title": title, "artist": artist, "type": "text"},
        ],
    )
    return sample


if __name__ == "__main__":
    downloaded_ids = {
        f.replace(".json", "")
        for f in os.listdir(META_DIR)
        if f.endswith(".json")
    }

    ds = load_dataset("vishnupriyavr/spotify-million-song-dataset")
    matched = ds["train"].filter(
        lambda s: safe_id(s["song"], s["artist"]) in downloaded_ids
    )
    matched.map(add_song)
