import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

song_collection = client.get_or_create_collection(
    name="songs", metadata={"hnsw:space": "cosine"}
)


embed = SentenceTransformer("all-MiniLM-L12-v2")


def embed_song(sample):
    song_name = sample["song"]
    song_embed = embed.encode(sample["text"])
    song_collection.add(
        ids=[song_name],
        embeddings=[song_embed.tolist()],
        metadatas=[{"artist": sample["artist"]}],
    )
    return sample


if __name__ == "__main__":
    ds = load_dataset("vishnupriyavr/spotify-million-song-dataset")
    ds = ds["train"].train_test_split(test_size=0.2)
    small_train = ds["train"].select(range(20000)).map(embed_song)
