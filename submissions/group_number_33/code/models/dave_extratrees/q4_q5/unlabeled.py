"""Section 5: test whether confidence-based pseudo-labeling helps."""

import numpy as np
import pandas as pd

from .common import (
    DATA_DIR,
    OUTPUT_DIR,
    auroc,
    build_model,
    extract_features,
    load_data,
)


CONFIDENCE_THRESHOLD = 0.95


def main():
    data, train_indices, validation_indices = load_data()
    unlabeled = pd.read_csv(DATA_DIR / "unlabeled.csv", keep_default_na=False)

    # Remove exact overlaps with labeled data and repeated unlabeled passages.
    unlabeled = unlabeled[~unlabeled["text"].isin(set(data["text"]))]
    unlabeled = unlabeled.drop_duplicates("text").reset_index(drop=True)

    labels = data["label"].to_numpy()
    labeled_features = extract_features(data["text"])
    unlabeled_features = extract_features(unlabeled["text"])

    # First train the supervised baseline on the 8,000 training passages.
    baseline = build_model()
    baseline.fit(labeled_features[train_indices], labels[train_indices])

    baseline_auroc = auroc(
        baseline,
        labeled_features[validation_indices],
        labels[validation_indices],
    )

    # Use only the baseline's most confident predictions as pseudo-labels.
    ai_probability = baseline.predict_proba(unlabeled_features)[:, 1]
    confident = (ai_probability >= CONFIDENCE_THRESHOLD) | (
        ai_probability <= 1 - CONFIDENCE_THRESHOLD
    )
    pseudo_labels = (ai_probability[confident] >= 0.5).astype(int)

    combined_features = np.concatenate(
        [labeled_features[train_indices], unlabeled_features[confident]]
    )
    combined_labels = np.concatenate([labels[train_indices], pseudo_labels])

    # Train the same model again with the pseudo-labeled passages added.
    student = build_model()
    student.fit(combined_features, combined_labels)

    student_auroc = auroc(
        student,
        labeled_features[validation_indices],
        labels[validation_indices],
    )

    results = pd.DataFrame(
        [
            {"method": "supervised_only", "validation_auroc": baseline_auroc},
            {"method": "with_unlabeled", "validation_auroc": student_auroc},
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "unlabeled_results.csv", index=False)

    pd.DataFrame(
        {
            "id": unlabeled.loc[confident, "id"].to_numpy(),
            "pseudo_label": pseudo_labels,
            "ai_probability": ai_probability[confident],
        }
    ).to_csv(OUTPUT_DIR / "pseudo_labels.csv", index=False)

    print(results.to_string(index=False))
    print(f"Selected {confident.sum()} of {len(unlabeled)} unlabeled passages")
    print(f"AUROC change: {student_auroc - baseline_auroc:+.5f}")
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
