import torch
from transformers import ClapModel, ClapProcessor, pipeline
from facenet_pytorch import MTCNN


class ClapEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained("laion/larger_clap_music").to(
            self.device
        )
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_music")


    def process_text(self, lyrics):
        text_inputs = self.processor(
            text=lyrics,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        return text_inputs

    def process_audio(self, audio):
        audio_inputs = self.processor(
            audio=audio, sampling_rate=48000, return_tensors="pt"
        ).to(
            self.device
        )  # something like tokenizer
        return audio_inputs

    def encode_text(self, text_inp):
        with torch.no_grad():
            text_vector = self.model.text_projection(
                self.model.text_model(**text_inp).pooler_output
            ) # projection keyword make text and audio in the same space

        return text_vector

    def encode_audio(self, audio_inp):
        with torch.no_grad():
            audio_vector = self.model.audio_projection(
                self.model.audio_model(**audio_inp).pooler_output
            )
        return audio_vector


class Facial:
    def __init__(self):
        self.emotion_classifier = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
        )
        self.face_detector = MTCNN(keep_all=True, device="cpu", post_process=False)