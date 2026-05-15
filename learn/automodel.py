from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch


def sentiment_classify(raw_out):
    raw_out.tolist()
    sentiment_outs = []
    for pair_out in raw_out:
        if pair_out[0] < pair_out[1]:
            sentiment_outs.append("positive")
        else:
            sentiment_outs.append("negative")

    return sentiment_outs


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)  # need specific tokenizer for specific model

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Tokenize
inputs = tokenizer(
    [
        "The ceasefire held for a third consecutive day, though tensions remain high.",
        "The company missed earnings expectations but raised its full-year guidance.",
    ],
    return_tensors="pt",
    padding=True,
)
# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

raw_out = outputs.logits

config = AutoConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
print(model)
