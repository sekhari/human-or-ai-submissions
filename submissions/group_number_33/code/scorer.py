import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from preprocess import _data_file

def evaluate_model(model, X_validation, y_validation, print_results=True):
    # Generate predicted class labels
    y_pred = model.predict(X_validation)

    # Generate prediction probabilities for AUROC
    y_prob = model.predict_proba(X_validation)[:, 1]

    # Calculate metrics
    f1 = f1_score(y_validation, y_pred)
    accuracy = accuracy_score(y_validation, y_pred)
    auroc = roc_auc_score(y_validation, y_prob)

    # Print results
    if print_results:
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUROC: {auroc:.4f}")

    return f1, accuracy, auroc

def create_kaggle_submission(model, X_test, output_file):

    ids = pd.read_csv(_data_file("test.csv"))["id"]
    y_pred = model.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({"id": ids, "target": y_pred})
    submission.to_csv(output_file, index=False)
    print(f"Kaggle submission saved to {output_file}")