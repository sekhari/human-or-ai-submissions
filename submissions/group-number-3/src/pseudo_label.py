"""Learning from unlabeled data via pseudo-labeling.

The student is always the word+char TF-IDF LogReg (its learning curve was still
rising at 8k rows, so more data should help it). We try two teachers:

  self:      the student itself labels unlabeled.csv (classic self-training)
  style_gbm: the stylometric GBM labels unlabeled.csv (teacher-student)

For each teacher and confidence threshold t, unlabeled passages with
p >= t (-> AI) or p <= 1 - t (-> human) are added to the 8,000 training rows.
Everything is scored on the same fixed 2,000-row validation set and compared
with a control trained on the 8,000 labeled rows only.

    python src/pseudo_label.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from data import ROOT, get_split, load_splits, load_train, load_unlabeled
from models import style_gbm, tfidf_logreg

THRESHOLDS = [0.99, 0.95, 0.90]


def val_auc(model, va):
    return roc_auc_score(va["label"], model.predict_proba(va["text"])[:, 1])


def main(output_dir=None):
    """Keep Aarushi's teachers, thresholds and full-pipeline refitting."""
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    train, splits, unl = load_train(), load_splits(), load_unlabeled()
    tr, va = get_split(train, splits, "random_fold")

    rows, selections = [], {}

    control = tfidf_logreg().fit(tr["text"], tr["label"])
    control_auc = val_auc(control, va)
    rows.append({"teacher": "none (control)", "threshold": None, "n_added": 0,
                 "frac_ai_added": None, "val_auroc": control_auc})
    pd.DataFrame({"id": va["id"], "label": va["label"],
                  "p_ai": control.predict_proba(va["text"])[:, 1]}).to_csv(
                      output_dir / "pseudo_label_control_predictions.csv", index=False)
    print(f"Control (8,000 labeled rows only): val AUROC = {control_auc:.6f}\n", flush=True)

    teachers = {
        "self": control,
        "style_gbm": style_gbm().fit(tr["text"], tr["label"]),
    }

    for name, teacher in teachers.items():
        p = teacher.predict_proba(unl["text"])[:, 1]
        for t in THRESHOLDS:
            mask = (p >= t) | (p <= 1 - t)
            pseudo = pd.DataFrame({"text": unl.loc[mask, "text"],
                                   "label": (p[mask] >= 0.5).astype(int)})
            assert not set(unl.loc[mask, "id"]) & set(train["id"])
            selections[f"{name}_{t}"] = {
                "ids": unl.loc[mask, "id"].tolist(),
                "human_count": int(pseudo["label"].eq(0).sum()),
                "ai_count": int(pseudo["label"].eq(1).sum())}
            aug = pd.concat([tr[["text", "label"]], pseudo], ignore_index=True)
            student = tfidf_logreg().fit(aug["text"], aug["label"])
            auc = val_auc(student, va)
            rows.append({"teacher": name, "threshold": t, "n_added": int(mask.sum()),
                         "frac_ai_added": round(pseudo["label"].mean(), 3),
                         "val_auroc": auc})
            if name == "style_gbm" and t == 0.95:
                pd.DataFrame({"id": va["id"], "label": va["label"],
                              "p_ai": student.predict_proba(va["text"])[:, 1]}).to_csv(
                                  output_dir / "pseudo_label_teacher_predictions.csv", index=False)
            print(f"teacher={name:<10} t={t:<5} added={mask.sum():>4} "
                  f"({pseudo['label'].mean() * 100:4.1f}% AI)  val AUROC = {auc:.4f}  "
                  f"({auc - control_auc:+.4f} vs control)", flush=True)

    df = pd.DataFrame(rows)
    rounded = df.copy()
    rounded["val_auroc"] = rounded["val_auroc"].round(4)
    out = output_dir / "pseudo_label.csv"
    rounded.to_csv(out, index=False)
    (output_dir / "pseudo_label_details.json").write_text(json.dumps({
        "based_on": "Aarushi's pseudo_label.py at 0e09450; method unchanged",
        "feature_policy": "Refit the full TF-IDF student pipeline on labeled plus pseudo-labeled texts",
        "thresholds": THRESHOLDS, "runs_full_precision": rows, "selections": selections,
        "validation_rows": len(va), "teacher_fit_labeled_rows": len(tr),
        "selection_caveat": "Style teacher at 0.95 was selected from exploration on this same validation fold; this rerun is not independent confirmation."
    }, indent=2) + "\n")
    report_table = df[(df["teacher"] == "none (control)") |
                      ((df["teacher"] == "style_gbm") & df["threshold"].eq(.95))]
    report_table.to_csv(output_dir / "pseudo_label_report.csv", index=False)
    print(f"\nSaved {out}", flush=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    main(parser.parse_args().output_dir)
