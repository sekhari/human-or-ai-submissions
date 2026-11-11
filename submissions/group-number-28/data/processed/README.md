# Recreated local partitions

`python3 -m human_ai.prepare_data` creates `train.csv` (8,000 rows) and
`validation.csv` (2,000 rows) here, preserving raw text and labels. These files
are ignored by Git. The committed ID lists in `data/splits/` are authoritative.
All learned preprocessing must be fitted inside the training partition.

