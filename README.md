# Emotion-Driven Music Recommender

A web application that reads your facial emotions through the webcam and recommends a song that matches how you feel — served as a FastAPI backend with a browser-based player UI, packaged with Docker.

---

## Concept

The system bridges the gap between **how you feel** and **what you listen to**. Instead of picking music manually, the browser captures a short webcam session, the backend reads the emotional state from those frames, and a CLAP-based vector search returns the song whose lyrics *and* audio best match that emotion.

```
Browser (webcam frames) → FastAPI → Emotion Detection → LLM Rephrase
                                       ↓
                       CLAP text embedding → ChromaDB query → Best song
```

1. **Capture** — The frontend grabs frames from the user's webcam over a session and POSTs them as base64 to `/face`.
2. **Detect** — MTCNN finds faces in each frame; a HuggingFace emotion classifier scores them.
3. **Aggregate** — Per-frame scores are accumulated; the top emotions of the session are selected.
4. **Rephrase** — A local Mistral model (via Ollama) turns those emotion labels into a descriptive sentence.
5. **Embed & Search** — CLAP (`laion/larger_clap_music`) encodes the sentence into a 512-dim vector and queries a ChromaDB collection where every song is stored under two modalities — a text (lyrics) embedding and an audio embedding. Scores from both are averaged to pick the best match.
6. **Play** — The frontend renders an Apple-Music-style player with album art, lyrics, and a YouTube link for the matched song.

---

## Project Structure

```
ntust_project/
├── README.md                ← you are here
├── demo/                    ← the deployed application
│   ├── main.py              ← FastAPI app (lifespan loads CLAP + Facial; serves /face and frontend)
│   ├── Dockerfile
│   ├── docker-compose.yml   ← app + ollama services
│   ├── requirements.txt
│   ├── api/
│   │   └── interface/interface.py   ← Pydantic `FacialAnalyst` request model
│   ├── src/
│   │   ├── frame_processor.py       ← decode base64 frames, run face + emotion detection
│   │   ├── emotions.py              ← LLM rephrase, CLAP encode, ChromaDB query
│   │   ├── cam.py                   ← (legacy CLI webcam loop)
│   │   ├── embed.py                 ← (legacy sentence-transformer pipeline)
│   │   └── run.py                   ← (legacy CLI entry point)
│   ├── model/
│   │   └── trans_model.py           ← `ClapEncoder` and `Facial` model wrappers
│   ├── frontend/
│   │   ├── index.html               ← setup / session / loading / player views
│   │   ├── app.js
│   │   └── styles.css
│   ├── tools/
│   │   ├── scrape.py                ← download song audio via yt-dlp
│   │   ├── clap_encode.py           ← encode (lyrics, audio) pairs
│   │   └── create_db.py             ← build the `clap-song-collection` ChromaDB
│   └── data/                        ← (gitignored) audio/, metadata/, clap_db/
└── learn/                           ← experiments and notebooks
```

---

## Tools & Libraries

### Computer Vision
| Tool | Purpose |
|---|---|
| **facenet-pytorch** (MTCNN) | Multi-face detection on each submitted frame |
| **OpenCV** / **Pillow** | Frame decoding and conversion for model input |

### Emotion Recognition
| Tool | Purpose |
|---|---|
| **Hugging Face Transformers** | Emotion classification via [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection) |

### Language & Embeddings
| Tool | Purpose |
|---|---|
| **Ollama + Mistral** | Local LLM that rephrases the top emotion labels into one descriptive sentence |
| **CLAP** (`laion/larger_clap_music`) | Dual text/audio encoder — projects emotion sentences and song lyrics/audio into a shared 512-dim space |

### Vector Search
| Tool | Purpose |
|---|---|
| **ChromaDB** | Persistent store of `clap-song-collection`; every song has both a `_text` and `_audio` entry. The query averages the two distances to pick the best match. |

### Backend & Frontend
| Tool | Purpose |
|---|---|
| **FastAPI** + **Uvicorn** | `/face` endpoint receives the frame batch; static frontend served at `/` |
| **Pydantic** | `FacialAnalyst` request schema |
| Vanilla **HTML / CSS / JS** | Webcam capture, session timer, player UI |

### Data Pipeline (one-time)
| Tool | Purpose |
|---|---|
| **yt-dlp** | Downloads song audio for the collection (`tools/scrape.py`) |
| **librosa** | Loads audio at 48 kHz for CLAP audio encoding |
| **Hugging Face Datasets** | Source of song lyrics (Spotify Million Song Dataset) |

---

## How to Run

### With Docker (recommended)

```bash
cd demo
docker compose up --build
```

This starts two services:

- `app` — FastAPI on `http://localhost:8000` (serves both the API and the frontend)
- `ollama` — Ollama on `http://localhost:11434`

Pull the Mistral model into the ollama container the first time:

```bash
docker compose exec ollama ollama pull mistral
```

Then open `http://localhost:8000` in a browser, allow camera access, start a session, and wait for the recommended song.

### Locally without Docker

```bash
cd demo
pip install -r requirements.txt

# Ollama must be reachable; emotions.py POSTs to http://ollama:11434
# (rename to http://localhost:11434 for non-Docker runs)
ollama pull mistral && ollama serve

uvicorn main:app --host 0.0.0.0 --port 8000
```

The ChromaDB at `demo/data/clap_db/` is required — it ships with the project and contains the pre-encoded song collection.

---

## Rebuilding the Song Collection

If you want to swap in your own songs:

```bash
cd demo
python tools/scrape.py       # download audio via yt-dlp into data/audio/
python tools/clap_encode.py  # encode (lyrics, audio) pairs with CLAP
python tools/create_db.py    # write embeddings into data/clap_db/
```

Each song lands in ChromaDB twice — once with id `<song>_text` and once with `<song>_audio` — so the query at runtime can compare both modalities and average their distances.

---

## API

### `POST /face`

Request body:

```json
{
  "frames": ["data:image/jpeg;base64,...", "..."],
  "duration": 30
}
```

Response:

```json
{
  "top_emotions": ["happy", "calm", "excited", "neutral"],
  "song": "john_lennon_imagine"
}
```

The frontend uses the returned `song` id to look up metadata, album art, lyrics, and a YouTube URL for playback.

---

## Architecture Diagram

```
┌──────────────┐  base64 frames   ┌─────────────────────┐
│  Browser     │ ───────────────▶ │  FastAPI  /face     │
│  (webcam +   │                  │  (main.py)          │
│   player UI) │ ◀─────────────── │                     │
└──────────────┘   {emotions,     └──────────┬──────────┘
                    song}                    │
                                             ▼
                                  ┌─────────────────────┐
                                  │  MTCNN + Emotion    │
                                  │  Classifier         │
                                  └──────────┬──────────┘
                                             │ top emotions
                                             ▼
                                  ┌─────────────────────┐
                                  │  Ollama / Mistral   │
                                  │  (rephrase)         │
                                  └──────────┬──────────┘
                                             │ sentence
                                             ▼
                                  ┌─────────────────────┐
                                  │  CLAP text encoder  │
                                  │  (512-dim vector)   │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │  ChromaDB query     │
                                  │  text + audio avg   │
                                  └──────────┬──────────┘
                                             │ best song id
                                             ▼
                                       to frontend
```
