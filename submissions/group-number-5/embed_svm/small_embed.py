import pandas as pd
import ollama
import json
from tqdm import tqdm

PATH = "submissions/group-number-5/embed_svm/"

df = pd.read_csv(PATH + "train.csv")

embeddings = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    embedding = ollama.embed(
        model="all-minilm:22m",
        input=row["text"]
    )["embeddings"][0]

    embeddings[str(row["id"])] = {
        "embedding": embedding,
        "label": row["label"]
    }

with open(PATH + "small_embeddings.json", "w") as f:
    json.dump(embeddings, f)