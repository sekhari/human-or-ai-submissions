"""Section 4: measure performance as the training set grows."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import OUTPUT_DIR, auroc, build_model, extract_features, load_data


TRAINING_SIZES = [500, 1000, 2000, 5000, 8000]


def make_nested_samples(labels):
    """Create balanced samples where every smaller sample is in the next one."""
    rng = np.random.default_rng(12)
    human = rng.permutation(np.flatnonzero(labels == 0))
    ai = rng.permutation(np.flatnonzero(labels == 1))

    samples = {}
    for size in TRAINING_SIZES:
        half = size // 2
        samples[size] = np.sort(np.concatenate([human[:half], ai[:half]]))

    return samples


def plot_results(results):
    plt.figure(figsize=(7, 4))
    plt.plot(
        results["training_rows"], results["training_auroc"], "o-", label="Training"
    )
    plt.plot(
        results["training_rows"],
        results["validation_auroc"],
        "o-",
        label="Validation",
    )
    plt.xlabel("Number of training passages")
    plt.ylabel("AUROC")
    plt.title("Effect of training-data size")
    plt.xticks(TRAINING_SIZES)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=200)
    plt.close()


def main():
    data, train_indices, validation_indices = load_data()
    labels = data["label"].to_numpy()
    features = extract_features(data["text"])
    samples = make_nested_samples(labels[train_indices])

    results = []

    for size in TRAINING_SIZES:
        selected_indices = train_indices[samples[size]]
        model = build_model()
        model.fit(features[selected_indices], labels[selected_indices])

        training_auroc = auroc(
            model,
            features[selected_indices],
            labels[selected_indices],
        )
        validation_auroc = auroc(
            model,
            features[validation_indices],
            labels[validation_indices],
        )

        results.append(
            {
                "training_rows": size,
                "training_auroc": training_auroc,
                "validation_auroc": validation_auroc,
            }
        )

        print(
            f"{size:>4} rows: "
            f"train AUROC={training_auroc:.5f}, "
            f"validation AUROC={validation_auroc:.5f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
    plot_results(results)

    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
