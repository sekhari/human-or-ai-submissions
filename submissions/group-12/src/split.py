"""Create the fixed stratified 8,000 / 2,000 train/validation split required by the report.

Run once from the repo root:  python -m src.split
Writes splits/val_ids.csv (committed) so every teammate uses the same validation rows.
"""
from sklearn.model_selection import train_test_split

from .data import SPLIT_DIR, SPLIT_SEED, VAL_IDS_PATH, VAL_SIZE, load_train


def main():
    train = load_train()
    _, val = train_test_split(train, test_size=VAL_SIZE, stratify=train["label"], random_state=SPLIT_SEED)
    SPLIT_DIR.mkdir(exist_ok=True)
    val[["id"]].sort_values("id").to_csv(VAL_IDS_PATH, index=False)
    print(f"wrote {VAL_IDS_PATH.relative_to(SPLIT_DIR.parent)}: {len(val)} rows, "
          f"AI share {val['label'].mean():.3f}, seed {SPLIT_SEED}")


if __name__ == "__main__":
    main()
