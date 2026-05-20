import torch
from transformers import ClapModel, ClapProcessor


class ClapEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def process(self, lyrics, audio):
        text_inputs = self.processor(
        text=lyrics, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        audio_inputs = self.processor(audio=audio, sampling_rate=48000, return_tensors="pt").to(
            self.device
        )  # something like tokenizer
        return text_inputs, audio_inputs
    
    def encode(self,text_inp, audio_inp):
        with torch.no_grad():
            text_vector = self.model.text_projection(
                self.model.text_model(**text_inp).pooler_output
            )
            audio_vector = self.model.audio_projection(
                self.model.audio_model(**audio_inp).pooler_output
            )
        return text_vector, audio_vector