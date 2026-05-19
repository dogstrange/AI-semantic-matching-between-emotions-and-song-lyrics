import torch
from transformers import ClapModel, ClapProcessor
import numpy as np
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(
    device
)  # .to() tells pytorch where to compute and moves tensor to that device
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

model.eval()  # inference mode, disable dropout


def encode_clap(lyrics: str, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    audio, sr = librosa(audio_path, sr=480000, mono=True)
    # preprocess both modalities
    text_inputs = processor(text=lyrics, return_tensors="pt", padding=True).to(device)
    audio_inputs = processor(audios=audio, sampling_rate=48000, return_tensors="pt").to(
        device
    )  # return the audio frequency array

    with torch.no_grad():
        text_vector = model.get_text_features(**text_inputs)
        audio_vector = model.get_audio_features(**audio_inputs)

    # normalize for cosine similarity
    text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
    audio_vector = audio_vector / audio_vector.norm(dim=-1, keepdim=True)

    return (
        text_vector.cpu().numpy(),
        audio_vector.cpu().numpy(),
    )  # .cpu() converts back to cpu array
