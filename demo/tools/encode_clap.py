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
    audio, sr = librosa.load(audio_path, sr=48000, mono=True)
    # preprocess both modalities
    text_inputs = processor(text=lyrics, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    audio_inputs = processor(audio=audio, sampling_rate=48000, return_tensors="pt").to(
        device
    )  # return the audio frequency array

    with torch.no_grad():
        text_vector = model.text_projection(model.text_model(**text_inputs).pooler_output)
        audio_vector = model.audio_projection(model.audio_model(**audio_inputs).pooler_output)

    # normalize for cosine similarity
    text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
    audio_vector = audio_vector / audio_vector.norm(dim=-1, keepdim=True)

    return (
        text_vector.squeeze(0).cpu().numpy(),
        audio_vector.squeeze(0).cpu().numpy(),
    )  # .cpu() converts back to cpu array
