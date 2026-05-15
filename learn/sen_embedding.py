from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences to embed
sentences = [
    "I am feeling very happy today",
    "Today is a great and joyful day",
    "The stock market crashed this morning",
    "Investors lost billions in the market collapse",
    "I love eating pizza",
]

# Generate embeddings
embeddings = model.encode(sentences)
print(embeddings)

# Compare all pairs
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = util.cos_sim(
            embeddings[i], embeddings[j]
        )  # compute score between 2 vectors, return a tensor value between 1 to -1
        print(f"{sentences[i][:30]} <-> {sentences[j][:30]} : {score.item():.4f}")
