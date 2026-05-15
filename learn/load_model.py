from transformers import pipeline

model_path = "./lora_results/checkpoint-375"
sentiment_classifier = pipeline("text-classification", model=model_path)


output = sentiment_classifier("My dad told me to watch this movie")


def label_classifier(sentments_arr):
    class_arr = []
    for item in sentments_arr:
        if item["label"] == "LABEL_0":
            class_arr.append("negative")
        else:
            class_arr.append("positive")
    return class_arr


print(label_classifier(output))
