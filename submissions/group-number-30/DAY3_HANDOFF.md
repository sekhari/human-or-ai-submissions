# Day 3 handoff - 2026-11-08

> Historical daily record. Pending tasks, branch names and report paths below describe that day. For the completed submission and current report, use [README.md](README.md) and [FINAL_HANDOFF.md](FINAL_HANDOFF.md).

## Outcome and provenance

Day 3 analysis and the integrated report draft are implemented on
`day3-validation-report`, based on Day 2 PR #1's head
`9485ee8357c5a7ae7656451618a6fda288bf4eab`. Review PR:
https://github.com/Divyesh-Thirukonda/kagglecomp/pull/2

Neither `main` nor Day 2's branch was changed or merged. This preserves both
teammates' work. This workspace PR is **not** the required course-repository PR.

The saved Day 1 and Day 2 arrays have identical training IDs, validation IDs,
and order; source CSV hashes also match. Day 2's original warning that the split
was unverified is now resolved by an explicit check, without rewriting its records.

## What Day 1 and Day 2 actually supplied

- Day 1: ten candidates from four lexical approaches (word LR, character LR,
  ComplementNB, calibrated linear SVM), five learning sizes, four structural
  ablations, and a later 50/50 character + 28-feature word/style/POS blend.
  Original selected validation AUROC: 0.978162. Core character fallback: 0.970756.
- Day 2: confidence pseudo-labeling (95 selected: 12 human, 83 AI), no AUROC
  improvement (0.970822 -> 0.970763); a **different** 11-feature compact-style
  blend (0.971276). It is not the original POS blend. Original Q3/Q5 results,
  notebook and run record remain unchanged.
- Runtime correction: the original Day 2 extra 0.87 seconds excludes word/style
  preprocessing. Do not describe it as total additional pipeline training time.
- Day 2's notebook did not populate the shared report source, commit its NPZ
  predictions, or provide a portable CLI. Day 3 integrated its recorded answers
  and added a replay CLI using the canonical saved split.

## Day 3 measured results

The final recommendation retains the original 28-feature blend, without tuning
new weights, thresholds, models or feature families on Day 3.

| Measurement | Result |
| --- | ---: |
| Training AUROC (8,000 rows) | 0.9989261875 |
| Validation AUROC (2,000 rows) | 0.978174 |
| Train-validation gap | 0.0207521875 |
| Validation log loss | 0.2978200992 |
| True negatives / false positives at 0.5 | 961 / 39 |
| False negatives / true positives at 0.5 | 111 / 889 |
| <150 whitespace words: n / AUROC | 380 / 0.968739 |
| 150-249 words: n / AUROC | 1,049 / 0.978170 |
| >=250 words: n / AUROC | 571 / 0.986630 |

The tagger resource hash matches Day 1 exactly. Package versions differ;
probabilities differ by up to 0.0158744812, so this is close ranking reproduction,
not identical probabilities. The original 18 tests and two added serialization-failure checks passed (20 total). `requirements-day3.txt`
records this actually executed environment; original requirements are preserved.

The portable Day 2 replay selected the same 12/83 pseudo-labels and confirmed its
qualitative conclusions: supervised 0.970838, pseudo-label 0.970816, compact blend
0.971245. The replay times feature extraction and fitting (20.22 seconds for the
character baseline, 26.88 seconds for both blend fits on this run). Timings are
hardware/load dependent; concurrent work was running, so they are not benchmarks.

## Error analysis (Q4)

Threshold fixed at P(AI)>=0.5. Deterministically select five highest-probability
FPs and five lowest-probability FNs, not five arbitrary favorable examples.

- FP IDs: 5908, 483, 3519, 6921, 2349.
- FN IDs: 8309, 4110, 5603, 1036, 1602.
- `results/day3/reviewed_error_cases.csv` gives probabilities, observations and
  possible failure explanations. These are AI-assisted close-reading hypotheses,
  **not causal attributions or independently human-reviewed conclusions**.
- Observed illustrative patterns include human recipe/advice passages and AI
  reference-like or personal/forum prose. Do not generalize their frequency to
  all errors; these are extreme-score examples. Do not externally fact-check or
  label test passages as a shortcut.

## Runnable commands

From `submissions/group-number-30/`, with authorized files in `../../data/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-day3.txt
.venv/bin/python -m nltk.downloader -d ../../data/nltk_data averaged_perceptron_tagger_eng
export NLTK_DATA="$PWD/../../data/nltk_data"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python train.py --train ../../data/train.csv --config results/recommended_config.json --results results
.venv/bin/python analyze_day3.py --train ../../data/train.csv
.venv/bin/python run_day2.py --data ../../data --out artifacts/day2_replay
.venv/bin/python train.py --train ../../data/train.csv --config results/recommended_config.json --full-data --model-out artifacts/day3_final.joblib
.venv/bin/python predict.py --test ../../data/test.csv --sample-submission ../../data/sample_submission.csv --model artifacts/day3_final.joblib --out artifacts/submission_day3.csv
```

Do not overwrite results: the repository does not include the saved validation
model. For a fresh refit, create a new output folder and copy `results/split.json`
there first. Use `--results <fresh_folder>` for `train.py`, then pass its prediction
path to `analyze_day3.py`. For the Day 2 replay, choose a new `--out` directory if
`results/day3/day2_replay/replay.json` exists. Model and submission paths also must
be new. Old teammate records remain evidence, not silently replaced measurements.

Report source: `report/main.tex` **inside the team folder**. The old root report
is preserved; the integrated team-local report is now the review version.

```bash
cd report
mkdir -p output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf -jobname=Group30_Report_Day3 main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf -jobname=Group30_Report_Day3 main.tex
```

## Files and final output

- `analyze_day3.py`: exact-ID/label checks, metrics, fixed bins, error selection.
- `run_day2.py`: portable replay of the teammate's notebook experiment.
- `results/day3/`: aggregate metrics, error IDs/hypotheses and replay results.
- `results/day3/verification.json`: runtime identity, full-training evidence,
  probability/file checks and explicit submission status.
- `report/main.tex`, `report/images/learning_curve.pdf`: assembled report source.
- `report/output/pdf/Group30_Report_Day3.pdf`: review draft; team must review it.
- Local only: `artifacts/submission_day3.csv` (3,000 unique test IDs, two columns,
  all probabilities in [0,1], template order verified).
- Local models are regenerable and excluded from Git. No competition passages,
  original data files, credentials or model weights are included in the PR.

## Remaining real-world submission steps

1. Shenye and teammates review the error cases, model choice, contributions and
   AI disclosure. Do not claim personal verification that has not occurred.
2. Confirm actual Kaggle team spelling/settings. Submit the generated CSV on
   Kaggle and record its real public score; private score remains unavailable.
3. Create/use the team's public fork of `sekhari/human-or-ai-submissions`, preserve
   substantive team history and open **one course PR** with changes only under
   `submissions/group-number-30/`. Put its real URL on the report title page and
   Q6.1. The private workspace is not a fork and PR #1/#2 cannot replace this.
4. Update pending online fields and the honest human-review disclosure, compile
   the PDF again and upload it to the course portal. Confirm the portal deadline.

No Kaggle upload, public score, private score or course-portal submission is claimed.

## Serialization verification follow-up

One secondary full-data artifact had an incomplete compressed stream on readback.
It was rejected; it never supplied the submission probabilities. The initial
full-data artifact remained readable. `train.py` now writes to a temporary file,
loads it to verify training scope/count, and only then publishes the final model
path. Two fixture tests cover success and an intentionally incomplete write.
A separate sequential full-data refit is checked against the original predictions;
its result is recorded in `results/day3/verification.json`.

Final sequential full-data reproduction succeeded: 3,000 IDs match and maximum probability difference is 0.0; the two CSVs are byte-identical. Twenty tests passed.
