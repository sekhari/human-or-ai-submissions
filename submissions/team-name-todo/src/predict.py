"""Write the Kaggle submission file."""
import argparse
import joblib, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--test", default="data/test.csv"); ap.add_argument("--model", default="model.joblib")
ap.add_argument("--out", default="submission.csv"); a = ap.parse_args()
t = pd.read_csv(a.test)
pd.DataFrame({"id": t.id, "label": joblib.load(a.model).predict_proba(t.text)[:, 1]}).to_csv(a.out, index=False)
