# the shared validation split

import argparse
from src.data import SEED, create_val_split, get_train_val, load_data

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", default="data/train.csv")
  parser.add_argument("--seed", type=int, default=SEED)
  args = parser.parse_args()

  full_df = load_data(args.train)
  create_val_split(full_df, seed=args.seed)

  train_df, val_df = get_train_val(full_df)

  print(f"Train: {len(train_df)} rows, {train_df['label'].mean():.1%} AI")
  print(f"Val:   {len(val_df)} rows, {val_df['label'].mean():.1%} AI")

if __name__ == "__main__":
  main()