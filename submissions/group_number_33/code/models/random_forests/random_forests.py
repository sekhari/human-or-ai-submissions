from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_data, tfidf_data_char, tfidf_data_combo, word2vec, sentence_embeddings
from scorer import create_kaggle_submission, evaluate_model

def random_forest(X_train, y_train, random_state=12, max_depth=None, n_estimators=100):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# f1 = []
# accuracy = []
# auroc = []
# X_train, y_train, X_validation, y_validation, X_test = preprocess_data()
# for j in range(1, 200):
#     model = random_forest(X_train, y_train, random_state=12, max_depth=None, n_estimators=j)
#     if j % 10 == 0:
#         print_results = True
#     else:
#         print_results = False
#     f1_score, acc, auc = evaluate_model(model, X_validation, y_validation, print_results=print_results)
#     f1.append(f1_score)
#     accuracy.append(acc)
#     auroc.append(auc)


# # F1
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(f1)), f1)
# plt.xlabel("Number of Estimators (i)")
# plt.ylabel("F1 Score")
# plt.title("F1 Score vs Number of Estimators")
# plt.show()

# # Accuracy
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(accuracy)), accuracy)
# plt.xlabel("Number of Estimators (i)")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs Number of Estimators")
# plt.show()

# # AUROC
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(auroc)), auroc)
# plt.xlabel("Number of Estimators (i)")
# plt.ylabel("AUROC")
# plt.title("AUROC vs Number of Estimators")
# plt.show()


X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization=tfidf_data_combo)
model = random_forest(X_train, y_train, random_state=12)
evaluate_model(model, X_validation, y_validation)
create_kaggle_submission(model, X_test, "submission_random_forest_tfidf_combo.csv")
