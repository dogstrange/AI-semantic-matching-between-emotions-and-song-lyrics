from transformers import AutoTokenizer

# Load the tokenizer for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize a sentence
result = tokenizer(
    ["I love my job", "I love my job but sometimes it can be really overwhelming"],
    padding=True,
)

print(f"input ids: {result['input_ids']}")
print(f"result: {result['attention_mask']}")
print(f"token type id: {result['token_type_ids']}")
