# Fixed evaluation partition — version 1

Commit `train_ids.csv`, `validation_ids.csv`, and `manifest.json` together.
They contain no passages or labels. Every controlled experiment uses these IDs.
The split is stratified: 4,000/class train and 1,000/class validation.

`human_ai/prepare_data.py` documents and implements the initial split algorithm.
It verifies existing files instead of silently replacing them. The manifest
checks the original `train.csv` bytes and both ID files using SHA-256.
If a hash differs, first check for a different download or modified line endings;
do not regenerate an easier validation partition.

The initial audit checks exact text duplicates after whitespace normalization.
It does not establish absence of near-duplicates or shared sources. Any split
change requires team agreement, a new split version, and rerunning comparisons.

