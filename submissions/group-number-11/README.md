# Group Number 11: Human or AI?

**Ansh Raj Bhargava (ab3396), Yilin Gao (yg695), Yingzi Gu (yg667)**

Kaggle team: **Group Number 11** · Submission branch: **submission/group-number-11**

[Final team branch](https://github.com/hisylvia66/human-or-ai-submissions/tree/submission/group-number-11) · [Course PR #13](https://github.com/sekhari/human-or-ai-submissions/pull/13)

## Final result and submission materials

The final model, **medium_v4**, achieved **0.98921 Kaggle public AUROC** (team-reported, November 11, 2026) and **0.999822 validation AUROC** on the shared 2,000-row holdout. Private results remain unknown. Model selection is complete.

| Material | Location |
| --- | --- |
| Submission-ready report | [report/main.pdf](report/main.pdf) |
| Editable LaTeX | [report/main.tex](report/main.tex) |
| Kaggle probabilities | `output/submission.csv` (local) |
| Overleaf project | `output/group11_overleaf.zip` (local) |
| PDF, CSV and Overleaf handoff | `output/Group_Number_11_final_materials.zip` (local) |
| Locked version and checks | [current_best.json](results/optimization/current_best.json), [delivery_manifest.json](results/delivery_manifest.json) |

The scored CSV SHA256 is `ac75c035ad713459781168a1f475ef14bedff26e59657742f438391c70ddeef6`. `output/submission_medium_v4.csv` is the same scored file. CSVs, input data, checkpoints and intermediate caches are excluded from Git; reproduce them with the commands below.

## Setup and reproduction

Run commands from the repository root. The tested environment is **Python 3.14.2, macOS arm64, 24 GiB RAM**; Apple MPS accelerates local language models. CPU fallback is supported but slower. On macOS, install the LightGBM runtime with `brew install libomp`. Tectonic is optional for rebuilding the PDF.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r submissions/group-number-11/requirements-optimization.txt
```

Place the original `train.csv`, `unlabeled.csv`, `test.csv` and `sample_submission.csv` together in a local data directory. This command trains/evaluates the components, extracts language features, and writes the final submission. It also rebuilds the report and handoff when Tectonic is available.

```bash
bash submissions/group-number-11/reproduce_final.sh \
  cornell-tech-cs-5785-combined-2026-ai-vs-human-text
```

Generic pretrained weights are downloaded from pinned Hugging Face revisions. Full reproduction is substantially slower than reblending cached predictions. After component inference, these stable entrypoints regenerate and verify the final CSV:

```bash
python submissions/group-number-11/predict_final.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --out submissions/group-number-11/output/submission.csv
python submissions/group-number-11/verify_final.py \
  --data cornell-tech-cs-5785-combined-2026-ai-vs-human-text \
  --require-scored-hash
```

Prediction reads cached component outputs and recorded weights; it does not retune or read training/validation labels. Verification checks all 3,000 test IDs, sample order, finite scores in [0, 1], CSV identity, validation labels and AUROC. The original scored file passed the strict hash check. Fresh retraining on a different platform can produce numerical differences; omit `--require-scored-hash` when checking such a new run. Kaggle receives continuous `id,label` scores, with `label` representing P(AI).

## Method and validation

The final score is a 75%/25% logit blend of the neural hybrid and a second language-statistics detector. The hybrid combines 90% small-GPT-2/style scores and 10% locally fine-tuned RoBERTa scores in logit space. The second detector uses small/medium GPT-2 statistics, style features, LightGBM and label-free score alignment within formatting groups. [Optimization notes](docs/OPTIMIZATION.md) record model revisions, settings and the measured public-score sequence, including the rejected candidate.

All required controlled experiments use the same stratified **8,000/2,000** split, seed **20261023**. Representations/classifiers fit training rows only during evaluation. Final tree components refit 10,000 labeled rows; RoBERTa retains its held-out-selected 8,000-row checkpoint (seed **20261111**). Supplementary split seeds are 20261024–20261026. The [baseline notes](docs/BASELINE.md) document model comparisons, hyperparameter grids, learning curves and the unlabeled-data experiment. Repeated holdout/public selection and distribution shift limit claims about private performance.

## Repository map

| Directory / file | Purpose |
| --- | --- |
| `predict_final.py`, `verify_final.py`, `reproduce_final.sh` | Stable final entrypoints, matching the report |
| `src/` | Training, features, prediction, analysis and report/packaging implementation |
| `scripts/` | Reference baseline, language_v2 and neural_v3 reproduction runners |
| `docs/` | Baseline protocol and optimization history |
| `results/` | Committed experiment evidence; [index](results/README.md) |
| `report/` | Final PDF, LaTeX and referenced figures |
| `report_metadata.json` | Team roles, public-score records and report settings |
| `requirements-optimization.txt` | Pinned final-model dependencies |
| `requirements.txt`, `environment.txt` | Original baseline environment |
| `artifacts/`, `output/` | Local models/caches and generated deliveries; ignored by Git |

To regenerate the report and packages after an intentional content change:

```bash
python submissions/group-number-11/src/build_report.py
(cd submissions/group-number-11/report && tectonic main.tex)
python submissions/group-number-11/src/package.py
python submissions/group-number-11/src/package_delivery.py
```

## Team contributions and AI disclosure

Yilin Gao led Q1 (model development); Ansh Raj Bhargava led Q3/Q5 (unlabeled data and improvements); Yingzi Gu led Q2/Q4 (learning curves and error analysis). All three also trained models and explored configurations. The final submission is consolidated on `submission/group-number-11` in `hisylvia66/human-or-ai-submissions`. Teammates should pull this branch before editing and push reviewed contributions here. `Group11retry` retains the development history, which is also included in the final merge.

OpenAI Codex assisted with implementation, local experiment execution, diagnostics and report drafting. All authorship supervision uses supplied labels. Generic GPT-2/RoBERTa weights run locally; no hosted prediction API or external authorship labels are used. Automated checks cover data hashes, split/label alignment, metrics, output schema and scored-file identity. The team should review the code and interpretations before course submission. Incremental development history is preserved in `git log --oneline -- submissions/group-number-11`.

Yilin Gao reports using OpenAI ChatGPT for baseline implementation, model comparison/tuning, interpretation, reproducibility/GitHub setup and Q1 drafting. Ansh Raj Bhargava reports using Claude (Anthropic) for Q3/Q5 experiment code, Kaggle/notebook and GitHub/reproducibility setup, interpretation and drafting, and running his experiments on Kaggle. Both reviewed the code, experimental outputs and report content before incorporating them. Their full individual disclosures are recorded verbatim in `report_metadata.json` and report Q6.3; they also cover earlier work retained in the merged team history.
