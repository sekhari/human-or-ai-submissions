"""Locked protocol and the model settings used in the report.

The validation split is the course holdout: 2,000 stratified rows from
train.csv, seed 20261023. Classifier seeds and hyperparameters are the
settings of the five models we kept (three single models, two ensembles).
"""

SPLIT_SEED = 20261023
MODEL_SEED = 42
VAL_SIZE = 2000
N_INNER_FOLDS = 3
LEARNING_CURVE_SIZES = (500, 1000, 2000, 5000, 8000)
DECISION_THRESHOLD = 0.5

BPE_VOCAB_SIZE = 30_000
BPE_MIN_FREQUENCY = 2
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 500_000

# Locked from the 3-fold search on the 8,000-row partition (split seed 20261023).
# The 2,000-row holdout was not used for this choice. Full rows are in results/metrics.json.
# SGD alpha: 1e-5 -> 0.9468, 5e-5 -> 0.9619, 1e-4 -> 0.9631, 5e-4 -> 0.9605.
# Logistic C: 1 -> 0.9522, 4 -> 0.9590, 16 -> 0.9618.
# GBM grid: 0.9886 to 0.9894; the locked row was 0.9893.
# Neighbors beat 5e-5 and C=4 by at most 0.003, so the submitted models keep those values.
SGD_ALPHA = 5e-5
SGD_ALPHAS = (1e-5, 5e-5, 1e-4, 5e-4)
LR_C = 4.0
LR_CS = (1.0, 4.0, 16.0)

GBM_LEARNING_RATE = 0.05
GBM_MAX_LEAF_NODES = 15
GBM_MIN_SAMPLES_LEAF = 40
GBM_L2 = 1.0
GBM_MAX_ITER = 300
GBM_GRID = (
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 0.1},
)

FINAL_MODEL = "ens_top2"
