import chromadb

# In-memory client — no persistence
client = chromadb.Client()

# Create a collection using cosine similarity
collection = client.create_collection(name="songs", metadata={"hnsw:space": "cosine"})

# Insert song vectors with metadata
collection.add(
    ids=["song_1", "song_2", "song_3"],
    embeddings=[
        [0.1, 0.2, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.3, 0.85],
    ],
    metadatas=[
        {"title": "Numb", "artist": "Linkin Park", "mood": "sad"},
        {"title": "Happy", "artist": "Pharrell", "mood": "joyful"},
        {"title": "Let Her Go", "artist": "Passenger", "mood": "melancholic"},
    ],
)  # each index is one sample

# Query with an emotion vector, return top 2 matches
results = collection.query(query_embeddings=[[0.1, 0.25, 0.88]], n_results=2)

print(results["ids"])
print(results["metadatas"])
print(results["distances"])
