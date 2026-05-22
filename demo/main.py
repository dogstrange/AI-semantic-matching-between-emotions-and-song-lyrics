import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from interface.interface import FacialAnalyst
from contextlib import asynccontextmanager
from model.trans_model import ClapEncoder, Facial
from src.frame_processor import analyze_frames
from src.emotions import top_emotions, clap_open_song
import chromadb


@asynccontextmanager  # same concept: with open(file):
async def lifespan(app):
    # startup — load once
    app.state.encoder = ClapEncoder()
    app.state.facial = Facial()

    app.state.clap_collection = chromadb.PersistentClient(
        path=os.path.join(os.path.dirname(__file__), "data", "clap_db")
    ).get_collection("clap-song-collection")

    yield  # server running

    # shutdown — cleanup if needed


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/face")
def analyze(
    req: Request, payload: FacialAnalyst
):  # fastapi know payload is in req because its FacialAnalyst interface
    frames = payload.frames
    facial = req.app.state.facial
    encoder = req.app.state.encoder

    analysis = analyze_frames(facial, frames)
    emotion_words = top_emotions(analysis, top_n=4)
    song = clap_open_song(encoder, emotion_words)

    out = {"top_emotions": emotion_words, "song": song}
    return out


app.mount(
    "/", StaticFiles(directory="frontend", html=True), name="frontend"
)  # StaticFiles serves every files that index.html reference from and sent additional get request
