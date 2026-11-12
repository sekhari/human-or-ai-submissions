import json
import numpy as np
from sklearn.decomposition import PCA

PATH = "submissions/group-number-5/embed_svm/"

with open(PATH + "embeddings.json") as f:
    data = json.load(f)

ids = list(data.keys())
X = np.array([data[i]["embedding"] for i in ids])

pca = PCA(n_components=X.shape[1]//2, svd_solver="full")
X_reduced = pca.fit_transform(X)

smaller_embeddings = {
    id_: {
        "embedding": embedding.tolist(),
        "label": data[id_]["label"]
    }
    for id_, embedding in zip(ids, X_reduced)
}

with open(PATH + "smaller_embeddings.json", "w") as f:
    json.dump(smaller_embeddings, f)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_reduced.shape[1]}")
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.4f}")