import numpy as np
import librosa


def encode_clap(
    model: object, lyrics: str, audio_path: str
) -> tuple[np.ndarray, np.ndarray]:
    audio, sr = librosa.load(
        audio_path, sr=48000, mono=True
    )  # load audio file return numpy array

    # preprocess both modalities
    text_inputs, audio_inputs = model.process(
        lyrics, audio
    )  # like tokenizer for this clap model
    text_vector, audio_vector = model.encode(text_inputs, audio_inputs)

    # normalize for cosine similarity
    text_vector = text_vector / text_vector.norm(dim=-1, keepdim=True)
    audio_vector = audio_vector / audio_vector.norm(dim=-1, keepdim=True)

    return (
        text_vector.squeeze(0).cpu().numpy(),
        audio_vector.squeeze(0).cpu().numpy(),
    )  # .cpu() converts back to cpu array
