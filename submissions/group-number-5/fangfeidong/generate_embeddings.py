import pandas as pd
import numpy as np
import requests

df = pd.read_csv("data/train_starter_2000.csv")

# print(df.shape)
# print(df["label"].value_counts())


texts = df["text"].fillna("").tolist()
embeddings = []

for i in range(0, len(texts), 20):
    batch = texts[i:i+20]

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "embeddinggemma-2", "input": batch},
        timeout=300
    )
    response.raise_for_status()

    embeddings.extend(response.json()["embeddings"])
    print("Processed:", len(embeddings), "/", len(texts))


X = np.array(embeddings)
np.save("data/starter_embeddings.npy", X)

print("Embedding shape:", X.shape)
