from pathlib import Path
import sys
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dave_extratrees import model
from preprocess import dave_145_stylization, preprocess_data
from scorer import evaluate_model

def n_estimator_expierement(random_state=12):
    print("\nExtra Trees n_estimator expierementation")


    X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=dave_145_stylization)

    best_score_model = None
    best_score = 0

    best_runtime = 0.0
    best_runtime_model = None

    # Store results for plotting
    estimator_values = []
    auroc_scores = []
    runtimes = []

    for n_estimators in [200, 250, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]:
        model_et = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=0.8,
            min_samples_leaf=2,
            max_depth=None,
            random_state=random_state,
            n_jobs=2,
            bootstrap=True,
            max_samples=0.8,
            class_weight='balanced_subsample'
        )

        start_time = perf_counter()
        model_et.fit(X_train, y_train)
        end_time = perf_counter()
        runtime = end_time - start_time


        f1, acc, auroc = evaluate_model(model_et, X_validation, y_validation, print_results=False)

        estimator_values.append(n_estimators)
        auroc_scores.append(auroc)
        runtimes.append(runtime)

        if  auroc > best_score:
            best_score = auroc
            best_score_model = model_et

        if runtime < best_runtime or best_runtime == 0.0:
            best_runtime = runtime
            best_runtime_model = model_et

    # Plot AUROC
    plt.figure(figsize=(12, 6))
    plt.plot(estimator_values, auroc_scores, label="Validation AUROC")

    plt.xlabel("Number of Estimators")
    plt.ylabel("AUROC")
    plt.title("Extra Trees: AUROC vs. Number of Estimators")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("extra_trees_auroc.png", dpi=300)
    plt.show()

    print(f"\nBest AUROC: {best_score:.5f}")
    print(f"Best n_estimators: {estimator_values[auroc_scores.index(best_score)]}")

    # Plot time
    plt.figure(figsize=(12, 6))
    plt.plot(estimator_values, runtimes, label="Training Time")

    plt.xlabel("Number of Estimators")
    plt.ylabel("Training Time (s)")
    plt.title("Extra Trees: Training Time vs. Number of Estimators")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("extra_trees_time.png", dpi=300)
    plt.show()

    print(f"\nBest Training Time: {best_runtime:.2f} seconds")
    print(f"Best n_estimators: {estimator_values[runtimes.index(best_runtime)]}")

    return best_score_model, best_score, best_runtime_model, best_runtime


def min_sample_leaf(random_state=12):
    print("\nExtra Trees min_samples_leaf expierementation")


    X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=dave_145_stylization)

    best_score_model = None
    best_score = 0

    # Store results for plotting
    estimator_values = []
    auroc_scores = []

    for min_samples_leaf in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        model_et = ExtraTreesClassifier(
            n_estimators=800,
            max_features=0.8,
            min_samples_leaf=min_samples_leaf,
            max_depth=None,
            random_state=random_state,
            n_jobs=2,
            bootstrap=True,
            max_samples=0.8,
            class_weight='balanced_subsample'
        )

        model_et.fit(X_train, y_train)

        f1, acc, auroc = evaluate_model(model_et, X_validation, y_validation, print_results=False)

        estimator_values.append(min_samples_leaf)
        auroc_scores.append(auroc)

        if  auroc > best_score:
            best_score = auroc
            best_score_model = model_et

    # Plot AUROC
    plt.figure(figsize=(12, 6))
    plt.plot(estimator_values, auroc_scores, label="Validation AUROC")

    plt.xlabel("min_samples_leaf")
    plt.ylabel("AUROC")
    plt.title("Extra Trees: AUROC vs. min_samples_leaf")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("extra_trees_auroc.png", dpi=300)
    plt.show()

    print(f"\nBest AUROC: {best_score:.5f}")
    print(f"Best min_sample_leaf: {estimator_values[auroc_scores.index(best_score)]}")

    return best_score_model, best_score

def max_features(random_state=12):
    print("\nExtra Trees max_features expierementation")


    X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=dave_145_stylization)

    best_score_model = None
    best_score = 0

    # Store results for plotting
    estimator_values = []
    auroc_scores = []

    for max_features in ["sqrt", 0.25, 0.50, 0.75, 1.0]:
        model_et = ExtraTreesClassifier(
            n_estimators=800,
            max_features=max_features,
            min_samples_leaf=2,
            max_depth=None,
            random_state=random_state,
            n_jobs=2,
            bootstrap=True,
            max_samples=0.8,
            class_weight='balanced_subsample'
        )

        model_et.fit(X_train, y_train)

        f1, acc, auroc = evaluate_model(model_et, X_validation, y_validation, print_results=False)

        estimator_values.append(max_features)
        auroc_scores.append(auroc)

        if  auroc > best_score:
            best_score = auroc
            best_score_model = model_et

    # Plot AUROC
    plt.figure(figsize=(12, 6))
    plt.plot(estimator_values, auroc_scores, label="Validation AUROC")

    plt.xlabel("max_features")
    plt.ylabel("AUROC")
    plt.title("Extra Trees: AUROC vs. max_features")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("extra_trees_auroc.png", dpi=300)
    plt.show()

    print(f"\nBest AUROC: {best_score:.5f}")
    print(f"Best max_features: {estimator_values[auroc_scores.index(best_score)]}")

    return best_score_model, best_score


best_score_model, best_score = max_features(random_state=12)
# best_score_model, best_score = min_sample_leaf(random_state=12)
# best_score_model, best_score, best_runtime_model, best_runtime = n_estimator_expierement(random_state=12)

X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=dave_145_stylization)

print(f"Best Score Model")
evaluate_model(best_score_model, X_validation, y_validation)





# print(f"Best Runtime Model")
# evaluate_model(best_runtime_model, X_validation, y_validation)