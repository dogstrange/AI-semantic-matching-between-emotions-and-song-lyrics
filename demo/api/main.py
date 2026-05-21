import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from fastapi import FastAPI, Request
from interface.interface import FacialAnalyst
from contextlib import asynccontextmanager
from trans_model import ClapEncoder, Facial
from frame_processor import analyze_frames
from emotions import top_emotions, clap_open_song
import chromadb


@asynccontextmanager  # same concept: with open(file):
async def lifespan(app):
    # startup — load once
    app.state.encoder = ClapEncoder()
    app.state.facial = Facial()
    
    app.state.clap_collection = chromadb.PersistentClient(
        path=os.path.join(os.path.dirname(__file__), "..", "data", "clap_db")
    ).get_collection("clap-song-collection")

    yield  # server running

    # shutdown — cleanup if needed


app = FastAPI(lifespan=lifespan)


@app.get("/")
def test():
    return {"message": "hellos"}


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

    return song


  