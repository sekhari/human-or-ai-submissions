# Human or AI? · Group Number 37

**Kaggle team:** Group Number 37

| Team member | NetID |
| --- | --- |
| Alexander Siladie | `aps335` |
| Cris Nicoglu | `cs2644` |
| Derrick Cardenas | `djc486` |

Our final submission combines a lexical and style reference, CatBoost on GPT-2 and GPT-2 Large features, DistilRoBERTa, and DeBERTa-v3-base. We normalize each component using quantiles fitted on a separate calibration set and blend them with weights 0.350, 0.234, 0.156, and 0.260. The public Kaggle AUROC was 0.97960.

## Setup

The full workflow was run on a Colab T4 with Python 3.13.16 and CUDA PyTorch 2.14.1. For Colab, open [Four_Method_Robust_Colab.ipynb](Four_Method_Robust_Colab.ipynb), select a T4 runtime, run all cells, and provide the competition ZIP when prompted. The notebook trains the current recipe and generates `submission_four_methods.csv`. The command-line prediction command below writes `submission.csv`.

For the command line, use a Python 3.13 environment with CUDA-enabled PyTorch installed. Download the supplied CSVs from Kaggle into `data/` at the repository root. From that root, run:

```bash
cd submissions/group-number-37
python -m pip install -r requirements-robust.txt
export USE_TF=0 USE_FLAX=0
```

The pretrained weights download on first use. Keep data, generated predictions, feature caches, and fitted weights outside Git.

## Evaluate on labeled validation data

```bash
python train_robust.py --data ../../data --output validation_model --validation --recipe current
python validate_robust.py --data ../../data --model validation_model
```

Seed 3737 sets the stratified 8,000/2,000 split and the model seeds. Seeds 9407 and 9413 divide the 8,000 development rows into 6,400 fitting rows, 800 calibration rows, and 800 selection rows. The evaluation script reports AUROC on the fixed 2,000-row audit for original text and formatting stress tests. Its output is `validation_model/audit_report.json`.

The earlier model comparisons and learning-curve and pseudo-labeling studies are in `experiments/`. Their findings are in [reports/FINDINGS.md](reports/FINDINGS.md). The older `train.py` and `predict.py` reproduce the three-method control. Its separate environment uses Python 3.12 and `requirements.txt`.

## Generate the Kaggle CSV

```bash
python train_robust.py --data ../../data --output final_model --recipe current
python predict_robust.py --model final_model --input ../../data/test.csv --output submission.csv --recipe current
```

Final fitting uses 9,200 labeled rows and reserves 800 for calibration. Prediction writes the original IDs and AI scores to `submission.csv`. The submitted CSV was generated using the fast reblend from the same fitted components. Full prediction implements the same four-component weights and can also score a new CSV with `id,text` columns. See [TEAM_HANDOFF.md](TEAM_HANDOFF.md) for the reblend workflow.

## Team contributions

- Alex developed the initial notebook and led the final four-component ensemble, expanded LM features, DeBERTa integration, calibration and robustness checks, T4 workflow, and accepted reblend with Codex assistance. His development commits are preserved in this branch.
- Cris developed the controlled experiments, learning curves, pseudo-labeling studies, dropout fix, domain checks, and earlier three-way blend.
- Derrick explored poetry-specific ranking methods and tested three Kaggle variants. Those experiments are documented in the report and Kaggle submission descriptions.

## AI assistance

OpenAI Codex assisted with coding, debugging, running experiments, explanations, and drafting and editing the report. As a team, we guided the modeling decisions, discussed compute constraints and robustness comparisons, and evaluated the Kaggle submission results. Automated checks covered data splits, prediction IDs and order, score ranges, and agreement between full scoring and the fast reblend. We used the checks and reported results to guide the final recipe. This matches Q6.3 in the report.
