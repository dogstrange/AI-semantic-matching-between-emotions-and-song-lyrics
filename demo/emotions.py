from embed import song_collection, embed
import requests
import webbrowser
import subprocess


def rephrase_emotions(emotions: list[str]) -> str:
    prmpt = f"You are describing the emotional journey of a person in short semtence.Given this sorted emotions from high to low in one session {', '.join(emotions)}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prmpt,
            "stream": False,
        },  # wait for ollama finish generate before returning
    )
    return response.json()["response"]


# def open_song(song, artist):
#     search = f"{song} {artist}".replace(" ", "+")
#     webbrowser.open(f"https://www.youtube.com/results?search_query={search}")


def play_song(title, artist):
    query = f"{title} {artist}"
    subprocess.run(
        ["mpv", f"ytdl://ytsearch1:{query}"]
    )  # ytdl:// means using ytdl protocol's


def find_and_open_song(emotion_words: list[str]):
    emotions_sentence = rephrase_emotions(emotion_words)
    print(f"input emotions sentence: {emotions_sentence}")

    embed_emotions = embed.encode(emotions_sentence)

    results = song_collection.query(
        query_embeddings=[embed_emotions.tolist()], n_results=1
    )

    song_name = results["ids"][0][0]
    artist = results["metadatas"][0][0]["artist"]
    distance = results["distances"][0]

    print(f"the distance: {distance}")
    play_song(song_name, artist)
    print(f"collection size: {song_collection.count()}")


if __name__ == "__main__":
    find_and_open_song(["stress", "sad"])
