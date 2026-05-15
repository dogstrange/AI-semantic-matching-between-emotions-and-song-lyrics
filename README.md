# Emotion-Driven Music Recommender

A real-time system that analyzes your facial emotions via webcam and recommends a song that matches how you feel — then plays it directly from YouTube.

---

## Concept

The core idea is to bridge the gap between **how you feel** and **what you listen to**. Instead of manually searching for music, the system reads your face over a short session, understands your emotional state, and finds a song whose lyrics semantically match that state.

The pipeline has three stages:

```
Webcam → Emotion Recognition → Semantic Song Search → Playback
```

1. **Capture** — Record your face for a fixed duration via webcam.
2. **Analyze** — Detect faces and classify emotions frame-by-frame; accumulate scores across the session.
3. **Rephrase** — Use a local LLM to turn the top emotions into a natural language description of your emotional journey.
4. **Search** — Encode that description into a vector and query a ChromaDB database of pre-embedded song lyrics to find the closest match.
5. **Play** — Stream the matched song from YouTube using `mpv` + `yt-dlp`.

Emotions are mapped to the **Circumplex Model of Affect** (e.g. `angry → stressed`, `surprise → excited`, `neutral → calm`) before being passed downstream, giving the language model richer vocabulary to work with.

---

## Project Structure

| File | Role |
|---|---|
| `run.py` | Entry point — orchestrates the full pipeline |
| `cam.py` | Webcam capture, face detection, per-frame emotion classification |
| `emotions.py` | LLM rephrasing + ChromaDB semantic search + playback |
| `embed.py` | One-time script to embed the Spotify Million Song dataset into ChromaDB |
| `song.py` | Helper to fetch lyrics via the Genius API for a personal song collection |
| `finetune_emo.py` | Fine-tuning script to adapt the sentence encoder on emotion-lyric pairs |
| `chroma_db/` | Persistent vector store (ChromaDB) holding song embeddings |

---

## Tools & Libraries

### Computer Vision
| Tool | Purpose |
|---|---|
| **OpenCV** (`cv2`) | Webcam capture and on-screen drawing |
| **facenet-pytorch** (MTCNN) | Fast multi-face detection in each frame |

### Emotion Recognition
| Tool | Purpose |
|---|---|
| **Hugging Face Transformers** | Emotion classification using [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection) |
| **Pillow** | Frame conversion for model input |

### Language & Embeddings
| Tool | Purpose |
|---|---|
| **Ollama + Mistral** | Local LLM that rephrases raw emotion labels into a descriptive sentence |
| **sentence-transformers** (`all-MiniLM-L12-v2`) | Encodes emotion sentences and song lyrics into comparable vectors |

### Vector Search
| Tool | Purpose |
|---|---|
| **ChromaDB** | Persistent vector database storing pre-embedded song lyrics; queried with cosine similarity |

### Dataset & Lyrics
| Tool | Purpose |
|---|---|
| **Hugging Face Datasets** | Loads the [Spotify Million Song Dataset](https://huggingface.co/datasets/vishnupriyavr/spotify-million-song-dataset) for bulk embedding |
| **lyricsgenius** | Fetches song lyrics via the Genius API for a personal collection |

### Playback
| Tool | Purpose |
|---|---|
| **mpv** | Media player used to stream the matched song |
| **yt-dlp** | YouTube backend (`ytdl://ytsearch1:`) used by mpv to resolve and stream audio |

### Fine-tuning
| Tool | Purpose |
|---|---|
| **PyTorch** | Training backend for the sentence encoder fine-tuning |
| **sentence-transformers** `CosineSimilarityLoss` | Trains the encoder to bring emotion descriptions closer to matching lyrics in vector space |

---

## How to Run

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with the Mistral model pulled (`ollama pull mistral`)
- `mpv` installed with `yt-dlp` support
- A `.env` file with your Genius API key (only needed for `song.py`):
  ```
  GENIUS_API_KEY=your_key_here
  ```

### 1. Embed the song dataset (one-time setup)

```bash
python embed.py
```

This downloads ~20,000 songs from the Spotify Million Song Dataset, encodes their lyrics, and stores them in `chroma_db/`.

### 2. Run the main application

```bash
python run.py
```

The webcam will open and analyze your face for **10 seconds**. After the session ends, it will find and play the best matching song.

---

## Architecture Diagram

```
┌─────────────┐     face crops      ┌──────────────────────┐
│  Webcam     │ ─────────────────▶  │  Emotion Classifier  │
│  (OpenCV)   │                     │  (Hugging Face)       │
└─────────────┘                     └──────────┬───────────┘
                                               │ emotion scores
                                               ▼
                                    ┌──────────────────────┐
                                    │  Circumplex Mapping  │
                                    │  + Top-N Selection   │
                                    └──────────┬───────────┘
                                               │ top emotions
                                               ▼
                                    ┌──────────────────────┐
                                    │  Ollama / Mistral    │
                                    │  (local LLM)         │
                                    └──────────┬───────────┘
                                               │ emotion sentence
                                               ▼
                                    ┌──────────────────────┐
                                    │  SentenceTransformer │
                                    │  (all-MiniLM-L12-v2) │
                                    └──────────┬───────────┘
                                               │ embedding vector
                                               ▼
                                    ┌──────────────────────┐
                                    │  ChromaDB            │
                                    │  (cosine similarity) │
                                    └──────────┬───────────┘
                                               │ matched song
                                               ▼
                                    ┌──────────────────────┐
                                    │  mpv + yt-dlp        │
                                    │  (YouTube stream)    │
                                    └──────────────────────┘
```
