# Classical Text Classification

Archived experiment: `classical_fixed2000_v002`.

Team: Sydney Patel (sjp294), Teddy Taylor (tft32), Zhenghao Zhou (zz2222).

Kaggle team: `group-number-34`.

The fixed split contains 8,000 training and 2,000 validation examples, with seed 42. All four models search C; the selected values are 16, 4, 1, and 4. The learning curve uses the combined model at C=16.

Use Python 3.12 and install `requirements.txt`. Place the competition data in `data/`.

```sh
python run_experiments.py --train data/train.csv --code-dir . --output-dir runs/reproduction --seed 42
python plot_results.py --results-dir runs/reproduction --output-dir figures
```

The results directory records the completed model comparison, parameter search, and learning curve.
