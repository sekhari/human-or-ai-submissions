import pandas as pd
import ollama
import json
from tqdm import tqdm

df = pd.read_csv("submissions/group-number-5/embed_svm/train.csv")

embeddings = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    embedding = ollama.embed(
        model="embeddinggemma-2",
        input=row["text"]
    )["embeddings"][0]

    embeddings[str(row["id"])] = {
        "embedding": embedding,
        "label": row["label"]
    }

with open("submissions/group-number-5/embed_svm/embeddings.json", "w") as f:
    json.dump(embeddings, f)