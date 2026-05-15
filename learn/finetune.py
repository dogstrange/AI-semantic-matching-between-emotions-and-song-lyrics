from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import numpy as np
from evaluate import load

"""
1. load the object we need (tokenizer, model, dataset, eval_matric)
2. train-test split the data
3. pass the datasets to the tokenizer since model only receive tokenized text
4. set up model training configuration
5. pass the model, the datasets, the eval functio and the configuration to the trainer for 'fine tuning'
"""

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Load and tokenize a small subset
dataset = load_dataset("imdb")


def tokenize(corpus):
    return tokenizer(
        corpus["text"], truncation=True, padding="max_length", max_length=512
    )


small_train = (
    dataset["train"]
    .select(range(1000))
    .map(tokenize, batched=True, remove_columns=["text"])
)  # have to pass the label as well, this is not inference
small_eval = (
    dataset["test"]
    .select(range(200))
    .map(tokenize, batched=True, remove_columns=["text"])
)  # only for evaluation doesn't update weight

# Load accuracy metric
metric = load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(
        predictions=predictions, references=labels
    )  # metric is from evaluate library


# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    compute_metrics=compute_metrics,
)

trainer.train()
