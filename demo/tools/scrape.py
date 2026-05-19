"""
Download audio files for every song already in the ChromaDB song_collection.
Lyrics are already encoded in the vector DB — no scraping needed.

Output layout:
  data/audio/<id>.mp3
  data/metadata/<id>.json   <- title, artist, youtube_query (for opening YouTube later)

Usage:
  python tools/scrape.py              # download all songs
  python tools/scrape.py --limit 200  # stop after 200 downloads
"""

import argparse
import json
import os
import re
import subprocess

import chromadb

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..")
CHROMA_PATH = os.path.join(DEMO_DIR, "chroma_db")
AUDIO_DIR = os.path.join(DEMO_DIR, "data", "audio")
META_DIR = os.path.join(DEMO_DIR, "data", "metadata")
PAGE_SIZE = 500

for d in (AUDIO_DIR, META_DIR):
    os.makedirs(d, exist_ok=True)


def safe_id(title: str, artist: str) -> str:
    raw = f"{artist}_{title}".lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def already_downloaded(song_id: str) -> bool:
    return os.path.exists(os.path.join(AUDIO_DIR, f"{song_id}.mp3"))


def download_audio(song_id: str, title: str, artist: str) -> bool:
    query = f"{artist} {title} official audio"
    out_path = os.path.join(AUDIO_DIR, f"{song_id}.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            f"ytsearch1:{query}",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", out_path,
            "--no-playlist",
            "--quiet",
        ],
        capture_output=True,
    )
    return result.returncode == 0


def iter_songs(collection):
    """Page through all songs in the collection without loading everything at once."""
    offset = 0
    while True:
        batch = collection.get(limit=PAGE_SIZE, offset=offset, include=["metadatas"])
        ids = batch["ids"]
        if not ids:
            break
        for title, meta in zip(ids, batch["metadatas"]):
            yield title, meta.get("artist", "")
        offset += len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max songs to download (0 = all)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("songs")
    total_in_db = collection.count()
    print(f"Found {total_in_db} songs in collection\n")

    collected = 0
    skipped = 0
    failed = 0

    for title, artist in iter_songs(collection):
        if args.limit > 0 and (collected + failed) >= args.limit:
            break

        song_id = safe_id(title, artist)

        if already_downloaded(song_id):
            skipped += 1
            continue

        print(f"[{collected + failed + skipped + 1}] {title} — {artist}", end="  ")

        ok = download_audio(song_id, title, artist)
        if ok:
            meta = {
                "id": song_id,
                "title": title,
                "artist": artist,
                "youtube_query": f"{artist} {title} official audio",
            }
            with open(os.path.join(META_DIR, f"{song_id}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            collected += 1
            print("ok")
        else:
            failed += 1
            print("failed")

    print(f"\nDone.  downloaded={collected}  already_had={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
