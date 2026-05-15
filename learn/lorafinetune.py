from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from evaluate import load

"""
1. load the object we need (tokenizer, model, dataset, eval_matric)
2. load lora config, wrap our model with it to freeze the original weights during fine tuning
3. train-test split the data
4. pass the datasets to the tokenizer since model only receive tokenized text
4. set up model training configuration
6. pass the model, the datasets, the eval functio and the configuration to the trainer for 'fine tuning'
"""

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# LoRA config (low rank adaptation)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,  # scale how in fine tune weight contributes to original weights
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
)  # output = original_weight + (lora_alpha / r) × (A × B)

# Wrap model with LoRA, peft = parameter effiecient fine tuning
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and tokenize
dataset = load_dataset("imdb")


def tokenize(corpus):
    return tokenizer(
        corpus["text"], truncation=True, padding="max_length", max_length=512
    )


small_train = (
    dataset["train"]
    .select(range(1000))
    .map(tokenize, batched=True, remove_columns=["text"])
)
small_eval = (
    dataset["test"]
    .select(range(200))
    .map(tokenize, batched=True, remove_columns=["text"])
)

# Metrics
metric = load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Training
training_args = TrainingArguments(
    output_dir="./lora_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    compute_metrics=compute_metrics,
)

trainer.train()
