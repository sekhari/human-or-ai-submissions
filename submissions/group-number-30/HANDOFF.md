# Day 1 handoff — November 5, 2026

> Historical daily record. Pending tasks, branch names and report paths below describe that day. For the completed submission and current report, use [README.md](README.md) and [FINAL_HANDOFF.md](FINAL_HANDOFF.md).

## Completed outcome

**Day 1 experimental and report work is complete on the actual supplied CSVs.** The user intentionally removed the initial provisional implementation; the code and conclusions here were rebuilt after inspecting the real data. No synthetic fixture score is used as competition evidence.

The Day 1 recommendation is an equal-weight soft blend of **character TF-IDF + logistic regression (C=3)** and **word TF-IDF + 28 style features + logistic regression (C=1)**. It scores **0.978162 validation AUROC** on the frozen 2,000-row holdout. The simpler character model remains a tested fallback at **0.970756**. Full-data refitting has completed, and a local submission file with all 3,000 test IDs has been generated and checked. **No Kaggle upload or public/private leaderboard score is claimed.**

## Day 1 exit evidence

| Requirement | Completed evidence |
| --- | --- |
| Data loading and fixed evaluation | `data_utils.py`, `results/split.json`; seed 20261023, 8,000/2,000 stratified rows, 4,000/4,000 training labels and 1,000/1,000 validation labels |
| At least three comparisons | Four approaches, ten measured candidates: word LR, character LR, ComplementNB and calibrated linear SVM |
| Tune at least two approaches | Three C values for each LR representation; three alpha values for NB; complete grid and timings in `results/comparison.json` |
| Five fixed-configuration learning sizes | 500/1,000/2,000/5,000/8,000; same character C=3 configuration and holdout; subset IDs and predictions saved |
| Baseline/configuration/predictions | `configs/baseline.json` is word LR C=1; `results/selected_config.json` is the selected core character model; every run has ID-labeled probability TSVs |
| Completed Q1/Q2 | Exact original questions and filled answers/tables/plot are inline in the workspace's root `report/main.tex`; the single report PDF is root `report/output/pdf/course-report-day1.pdf`. |
| Written handoff and next steps | This file, README, DATA_NOTES, HYPOTHESES and updated root TEAM_PLAN |

### Core comparison winners

| Approach | Best setting | Train AUROC | Validation AUROC |
| --- | --- | ---: | ---: |
| Word logistic regression | C=3.0 | 0.999817 | 0.941214 |
| Character logistic regression | C=3.0 | 0.999148 | 0.970756 |
| ComplementNB | alpha=1.0 | 0.982453 | 0.898849 |
| Calibrated linear SVM | C=1.0 | 1.000000 | 0.943227 |

Use the saved JSON for full precision and timings. Character-model learning-curve validation AUROCs are **0.875650, 0.923034, 0.948718, 0.965088, 0.970755** in increasing sample-size order. The 8,000-row curve differs from the comparison by 0.000001 because the nested sample has a different row order; its IDs comprise the same training partition. No hyperparameter was retuned for the curve.

## What the handwritten ideas showed

The actual training corpus mixes recipes, news, scientific text and informal posts. Adjective/positivity effects were weak; superlatives and first-person wording trended opposite the suggested direction. Repetition and formatting showed stronger differences but can reflect source artifacts. `DATA_NOTES.md`, `HYPOTHESES.md` and `results/training_feature_audit.json` preserve the evidence and tokenization definitions.

| Fixed C=1 ablation | Validation AUROC | Change versus word LR C=1 (0.935686) |
| --- | ---: | ---: |
| 28 style features alone | 0.906037 | −0.029649 |
| POS 1–3 grams | 0.947155 | +0.011469 |
| Cross-fitted class-conditional POS trigram surprise | 0.926211 | −0.009475 |
| Word TF-IDF plus style | 0.955052 | +0.019366 |

All four completed without recorded warnings. The surprise approach is an implemented statistical test of the next-word-type idea, not a neural network or a grammatical-error detector. Semantic “unnecessary adjective” and “poetry” judgments are not implemented or asserted.

After these results, we declared **one** equal-weight blend, without searching weights. It improved character-only AUROC by **+0.007406**, with an exploratory paired 95% bootstrap interval **[+0.003951, +0.011163]**. It met the recorded promotion heuristic. Its fit time was **40.54 seconds**, compared with **9.00 seconds** for the core character model. Its log loss was **worse** (0.297826 versus 0.281739), so the choice optimizes the competition's ranking metric, not probabilistic scoring. Log loss alone does not diagnose calibration. The interval reuses a selection holdout and does not account for all model search; it is not confirmatory evidence.

## Data diligence and limitations

Official download: `cornell-tech-cs-5785-combined-2026-ai-vs-human-text`, Data tab in the user's authenticated Chrome. Actual schema is `id,text,label` for training, `id,text` for test/unlabeled, and **`id,label`** for output. Starter is an exact subset, not extra labels. There are no blank fields, repeated IDs, exact/normalized duplicate texts, or exact cross-file text overlaps.

The all-pairs audit found no word-trigram Jaccard pairs at ≥0.70. Eight validation rows have moderate lexical overlap or shared 80-character openings with training. Keeping the prescribed primary holdout unchanged and excluding those eight only for a sensitivity check gives **0.978041** for the blend and **0.970627** for the character model on 1,992 rows. This does not prove complete source independence; domain/generator/source labels are unavailable. All learned preprocessing and class-conditioned statistics are fit on training only. The supplied unlabeled set has not been used for model fitting yet.

## Exact verified commands in this checkout

From the repository root:

```bash
source .venv/bin/activate
export NLTK_DATA="$PWD/data/nltk_data"
cd submissions/group-number-30
python -m unittest discover -s tests -v
python inspect_data.py --data ../../data --results results_inspection
```

To rerun the experiments while preserving the original evidence, use a fresh output directory:

```bash
mkdir -p results_reproduction
cp results/split.json results_reproduction/split.json
cp results/extension_plan.json results_reproduction/extension_plan.json
python inspect_data.py --data ../../data --results results_reproduction
python run_day1.py --data ../../data --results results_reproduction --bonus --blend
```

Edit the one canonical report at root `report/main.tex`; Q1/Q2 are inline. To regenerate only its figure from this team directory, use `python run_day1.py --results results --plot-only --figure-out ../../report/images/learning_curve.pdf`, then compile the root report as documented in README. Duplicate report/output trees and prose-generating scripts were removed; the blend experiment is now `run_day1.py --blend` (or `--blend-only` for existing component results). Only team-directory code belongs in the course PR; the report is submitted separately.

The compact `results/` keeps summaries, fixed split/configurations and `validation.predictions.tsv` for the recommended blend. `day1-evidence.zip` preserves all 65 original small result files, including every per-run prediction and learning subset ID, with a SHA256 manifest. README gives extraction commands. The consolidated `audit.json` includes the former source audit's additional sample-submission checks. Intermediate models and scratch files have a verified recovery archive outside the workspace; the actual full-data model, metadata and Kaggle CSV remain in `artifacts/`.

The actual completed full-data commands were:

```bash
python train.py --train ../../data/train.csv --config results/recommended_config.json \
  --full-data --model-out artifacts/day1_recommended.joblib
python predict.py --test ../../data/test.csv --sample-submission ../../data/sample_submission.csv \
  --model artifacts/day1_recommended.joblib --out artifacts/submission_day1_recommended.csv
```

Output paths are protected from overwrite; use new names for reruns. The sample's ID order is authoritative, and prediction verifies each ID and probability. For the simpler fallback, refit `results/selected_config.json` with separate output paths. Old validation model files were removed from the active workspace; their measured predictions remain in the evidence ZIP.

For a fresh checkout, follow README: Python **3.12.14**, pinned `requirements.txt`, official Kaggle download in root `data/`, and the disclosed NLTK English tagger in `data/nltk_data`. Data, weights and credentials are excluded from Git. Small results, split IDs, aggregate audits, validation probabilities and plots are included. `results/software_verification.json` records the final clean-source reproduction check and output hash.

## Development history and provenance

- `a4c2805`: new actual-data-based pipeline, audit, model comparison and structure implementations. Original core/curve/bonus measurements use implementation SHA `4e0916a5758e589d28672343c32d49ceacf3bf87d08ba5ed32c76943e519496b`.
- `70c8a61`: one adaptive blend plus stricter resume environment/artifact integrity checks. Blend and full-model artifacts use implementation SHA `9504096e81b9ee1a1119b6d131b2acffb75adb1d73c6d31847f468ccac53b049`.
- The subsequent handoff commit adds measured results, report/PDF, verification evidence and final documentation. Consult `git log` for that final revision.

Resume hardening did not change the original model fits. Old records intentionally cannot be reused with `--resume` under a different implementation hash. Use fresh output directories to reproduce them. The compact evidence folder is not a resumable working directory; original models are excluded from its archive. Consolidating the experiment runner did not change the scoring implementation fingerprint. The user is reviewing local changes; do not commit or push without their approval.

## Next person's first steps

1. Read Q1/Q2 in root `report/main.tex` and team `HYPOTHESES.md`; inspect the training examples/audit, and record your own verification for the AI disclosure. Earlier report notes are retained in the evidence ZIP.
2. Reproduce the recommended configuration on the **same split** using `python train.py --train ../../data/train.csv --config results/recommended_config.json --results results` (writes `results/verify/selected.*`), or a fresh directory with the copied split. Do not change the split to improve a score.
3. **Day 2 remains Q3/Q5/Q6.1:** run a controlled unlabeled-data experiment without letting validation data enter feature fitting or pseudo-label training; review the existing structural/blend evidence, probabilistic-scoring cost and generalization; make the final model recommendation and write those sections. The extra Day 1 work supplies evidence for Q5 but does not replace Day 2's review.
4. **Day 3 remains Q4/Q6.2/Q6.3 and assembly:** at least three length bins, five false positives and five false negatives with a stated threshold, human verification/disclosure, final report fields, course PR and Kaggle upload. Q0 and later report placeholders remain intentionally unfinished.
5. Confirm missing teammates' NetIDs, named shift owners and actual Kaggle team configuration. The required course name is Group Number 30; no team invitations or account settings were changed here. Kaggle displayed its close as November 11, 2026 at 10:59 p.m. Central; confirm course report deadlines separately.

Codex performed implementation, experiments, software checks and report drafting. The user supplied hypotheses and required data-first diligence. No human review is claimed for work teammates have not actually checked, and no hidden labels or external LLM passage predictions were obtained.
