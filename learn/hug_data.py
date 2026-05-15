from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def tokenize(corpus):
    return tokenizer(
        corpus["text"], truncation=True, padding="max_length", max_length=512
    )


dataset = load_dataset("imdb")
tokenize_data = dataset["test"].map(tokenize, batched=True)
split = tokenize_data.train_test_split(test_size=0.1)

print(split)
