# Original competition files (local, not committed)

Place `train.csv`, `unlabeled.csv`, `test.csv`, `train_starter_2000.csv`, and
`sample_submission.csv` here, unchanged. Each teammate obtains the competition
files through their own authorized access. The shared utilities also accept the
existing root directory `cornell-tech-cs-5785-combined-2026-ai-vs-human-text/`
as a fallback, so the original files need not be moved or duplicated.

Run `python3 -m human_ai.prepare_data` from the repository root. An explicit
`--raw-dir /path/to/files` overrides both default locations. Git tracks split IDs
and source hashes, not passages or labels. Do not put local absolute paths in
shared experiment configurations.

