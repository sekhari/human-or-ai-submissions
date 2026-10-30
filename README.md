# Human or AI? — Course Code Submissions

Cornell Tech · Applied Machine Learning · Kaggle competition code repository.

Teams submit **one pull request** against this repository with runnable code, setup instructions, and a meaningful commit history. Put the PR URL on your report title page and in the course submission portal.

> **This repository is for code submissions only.** Competition data lives on Kaggle. Do not commit large data files, model checkpoints, or anything that recovers hidden test labels.

## Repository

**https://github.com/sekhari/human-or-ai-submissions**

This repository is **public**. You do not need a collaborator invite. Fork it, work in your fork, and open a PR back here.

## How to submit (one PR per team)

### 1. Fork this repository

On GitHub, click **Fork** (top right) to create a copy under your own account. One fork per team is enough — add teammates as collaborators on *your* fork if you want shared push access.

### 2. Clone *your fork* and create a team branch

```bash
git clone git@github.com:<your-github-username>/human-or-ai-submissions.git
cd human-or-ai-submissions
git checkout -b submission/<kaggle-team-name>
```

Use your exact Kaggle team name (lowercase, hyphens instead of spaces). Example: `submission/cornell-detectives`.

Optional but recommended — keep the course repo as `upstream`:

```bash
git remote add upstream git@github.com:sekhari/human-or-ai-submissions.git
git fetch upstream
```

### 3. Add your code under `submissions/`

Create **one directory** for your team and put all project code there:

```text
submissions/<kaggle-team-name>/
  README.md           # required: how to set up and reproduce
  requirements.txt    # or environment.yml / pyproject.toml
  train.py            # or notebooks + scripts — your entrypoints
  predict.py          # command that writes submission.csv
  ...                 # other source files you need
```

See [`submissions/_example_team/`](submissions/_example_team/) for a minimal layout.

**Rules**

- Only add or modify files under `submissions/<your-team-name>/`.
- Do **not** edit other teams’ folders, this root `README.md`, or `.github/`.
- Do **not** commit `train.csv`, `test.csv`, `unlabeled.csv`, Kaggle cookies, API keys, or full model weight dumps larger than needed for grading.
- Keep commits incremental and meaningful (experiments, fixes, refactors). Do not squash away development history before review.

### 4. Document reproducibility in your team README

Your team `README.md` must include:

1. Team members (names + NetIDs) and Kaggle team name  
2. Python version and how to install dependencies  
3. Exact commands to train / evaluate on the labeled validation protocol  
4. Exact command that writes `submission.csv` for Kaggle  
5. Random seeds and any important config  
6. AI-tool disclosure summary (Claude / ChatGPT / Copilot / etc.), matching Q6.3 in the report  

### 5. Push to your fork and open the pull request

```bash
git add submissions/<kaggle-team-name>
git commit -m "Add <kaggle-team-name> competition code"
git push -u origin submission/<kaggle-team-name>
```

Then on GitHub, open a PR **from your fork** into this course repository:

- **Base repository:** `sekhari/human-or-ai-submissions`
- **Base branch:** `main`
- **Head repository:** your fork
- **Compare branch:** `submission/<kaggle-team-name>`
- **Title:** `[Submission] <kaggle-team-name> — <NetID1>, <NetID2>, ...`
- **Body:** short summary of the final approach + confirmation that `submission.csv` can be regenerated from the listed commands

GitHub’s “Contribute → Open pull request” button on your branch usually fills this in.

Copy the PR URL into:

1. The **GitHub PR URL** field on your report title page  
2. The course submission portal  

## What we grade from the PR (rubric §6)

| Item | Expectation |
|---|---|
| Accessible PR | Link opens; code is under your team folder |
| Entrypoints | Clear train/eval and predict commands |
| Dependencies | Installable environment documented |
| Seeds / repro | Commands and seeds recorded |
| Development history | A few substantive commits with lessons (Q6.2) |
| AI disclosure | Honest account of AI use and what you verified |

We inspect the **final code and its development history**, not only the last snapshot.

## Competition reminders

- Do not retrieve hidden labels from RAID or any external source.
- Do not use external LLM APIs to directly label or predict competition test passages.
- Laptop-friendly pipelines are sufficient for full credit; GPUs are optional.

## Questions

Post on the course Ed / Canvas channel, or email course staff.
