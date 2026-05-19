import torch
from transformers import ClapModel, ClapProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


class Load_model:
    def __init__(self, name: str):
        self.name = name
        if self.name == "CLAP":
            self.get_Clap()

    def get_Clap(self):
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
