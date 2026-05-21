import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from fastapi import FastAPI
from interface.interface import FacialAnalyst
from contextlib import asynccontextmanager
from trans_model import ClapEncoder, Facial

@asynccontextmanager  # same concept: with open(file):
async def lifespan(app):
    # startup — load once
    encoder = ClapEncoder()
    facial = Facial()
    yield  # server running

    # shutdown — cleanup if needed


app = FastAPI()


@app.get("/")
def test():
    return {"message": "hellos"}


@app.post("/face")
def analyze(payload: FacialAnalyst):
    pass
