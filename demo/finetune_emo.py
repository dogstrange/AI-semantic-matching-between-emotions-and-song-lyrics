from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("all-MiniLM-L6-v2")

# Your training pairs
examples = [
    InputExample(texts=["sad lonely tired", "song lyrics here"], label=1.0),  # similar
    InputExample(
        texts=["sad lonely tired", "happy upbeat lyrics"], label=0.0
    ),  # not similar
]

dataloader = DataLoader(examples, batch_size=16)
loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(dataloader, loss)], epochs=3)
model.save("./fine_tuned_model")
