# Emotion-Driven Music Recommendation System

A real-time application that detects your facial emotions via webcam and recommends a matching song using the CLAP (Contrastive Language-Audio Pretraining) model.

## How It Works

1. Webcam captures your face for a session (10 seconds)
2. A facial emotion classifier detects emotions frame-by-frame
3. Top emotions are rephrased into a natural sentence via a local LLM (Mistral via Ollama)
4. CLAP encodes the sentence into a 512-dim vector
5. The vector is queried against a ChromaDB collection of ~200 songs encoded by both lyrics and audio
6. The closest matching song is played via mpv (YouTube search)

## Tech Stack

- **Emotion detection**: `facenet-pytorch` (MTCNN) + HuggingFace `dima806/facial_emotions_image_detection`
- **Music embedding**: `laion/clap-htsat-unfused` (dual encoder: text + audio)
- **Vector database**: ChromaDB (cosine similarity, ~400 entries = 200 songs × 2 modalities)
- **LLM**: Mistral running locally via Ollama
- **Audio playback**: mpv with yt-dlp
- **Song dataset**: Spotify Million Song Dataset (HuggingFace) — lyrics + scraped audio

## Project Structure

```
demo/
├── run.py              # Entry point
├── cam.py              # Webcam emotion session
├── emotions.py         # CLAP query + song recommendation
├── embed.py            # Legacy sentence-transformer pipeline
├── model/
│   └── trans_model.py  # ClapEncoder class
├── tools/
│   ├── clap_encode.py  # Encode (lyrics, audio) pair
│   ├── create_db.py    # Build clap-song-collection in ChromaDB
│   └── scrape.py       # Download audio files via yt-dlp
└── data/               # (gitignored) audio/, metadata/, clap_db/
```

## Current State

- Data pipeline complete: 200 songs scraped and encoded into ChromaDB
- CLI pipeline works end-to-end: webcam → emotion → song
- **Next step**: wrap in a FastAPI backend + HTML/JS frontend for web deployment

## Production TODO

- [ ] `demo/api.py` — FastAPI endpoint `POST /analyze` wrapping the pipeline
- [ ] `demo/static/index.html` — browser UI with webcam preview and song result
- [ ] Fix `emotions.py` to return song metadata (title, artist) instead of calling mpv
- [ ] Dockerfile — containerize with model weights pre-downloaded
- [ ] Replace mpv playback with YouTube URL returned to frontend

## Local Setup

```bash
# Dependencies
pip install torch transformers librosa chromadb facenet-pytorch fastapi uvicorn

# Requires Ollama running locally with Mistral
ollama run mistral

# Run
cd demo
python run.py
```
