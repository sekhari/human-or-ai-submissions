"""Eight-family Q1 study with training-only coarse search and one refinement.

The original v1 runner/artifacts remain unchanged. All new evidence uses a new
prefix; sparse and dense representations have an explicit LR bridge control.
"""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import platform
import resource
import shutil
import time
import warnings

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT, SPLIT_DIR, load_partitions, raw_path, read_csv, sha256, write_csv
from .run_q1 import (bootstrap, clean_params, id_hash, predict_ai, provenance,
                     save_json, score, vectorizer)

HERE = Path(__file__).resolve().parent
NAMES = {"lr": "Logistic regression", "nb": "Multinomial NB", "svm": "Calibrated linear SVM",
         "tree": "Decision tree", "forest": "Random forest", "boost": "Histogram gradient boosting",
         "mlp": "Neural network (MLP)", "lr_svd": "Logistic regression (SVD control)"}


def load_plans():
    plan = json.loads((HERE / "q1_expanded_plan.json").read_text())
    original = json.loads((HERE / "q1_plan.json").read_text())
    return plan, original


def representation(family, plan):
    return plan["representations"]["by_family"][family]


def classifier(family, params, seed=42):
    p = dict(params)
    if family in ("lr", "lr_svd"):
        return LogisticRegression(solver="liblinear", max_iter=3000, tol=1e-5, random_state=seed, **p)
    if family == "nb":
        return MultinomialNB(force_alpha=True, **p)
    if family == "svm":
        return LinearSVC(dual="auto", tol=1e-5, max_iter=10000, random_state=seed, **p)
    if family == "tree":
        return DecisionTreeClassifier(random_state=seed, **p)
    if family == "forest":
        return RandomForestClassifier(n_jobs=1, random_state=seed, **p)
    if family == "boost":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **p)
    if family == "mlp":
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=20, tol=1e-4, random_state=seed, **p)
    raise ValueError(family)


def make_model(family, params, plan, original, seed=42):
    rep = representation(family, plan)
    steps = [("tfidf", vectorizer(original))]
    if rep != "sparse":
        steps.append(("svd", TruncatedSVD(n_components=128, n_iter=7, random_state=42)))
    if rep == "scaled_dense":
        steps.append(("scale", StandardScaler()))
    steps.append(("classifier", classifier(family, params, seed)))
    model = Pipeline(steps)
    if family == "svm":
        cv = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["calibration"])
        model = CalibratedClassifierCV(model, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
    return model


def signature(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def refinement(family, best):
    proposals = []
    def change(**kw):
        proposals.append({**best, **kw})
    if family in ("lr", "lr_svd", "svm", "nb"):
        key = "alpha" if family == "nb" else "C"
        for scale in (1/3, 3):
            change(**{key: best[key] * scale})
    elif family == "tree":
        leaf, depth = best["min_samples_leaf"], best["max_depth"]
        for value in (max(1, leaf//2), leaf*2):
            change(min_samples_leaf=value)
        for value in ((30, 60) if depth is None else (max(2, depth//2), depth*2)):
            change(max_depth=value)
    elif family == "forest":
        for value in (200, 400):
            change(n_estimators=value)
    elif family == "boost":
        change(max_iter=300)
        change(learning_rate=best["learning_rate"]/2)
        change(max_iter=300, learning_rate=best["learning_rate"]/2)
    elif family == "mlp":
        change(alpha=best["alpha"]/10)
        change(alpha=best["alpha"]*10)
        change(hidden_layer_sizes=[128])
    return proposals


def fit_record(model, x, y):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    notes = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
    return time.perf_counter()-start, notes


def prepare_folds(train, plan, original, study):
    text = [r["text"] for r in train]
    y = np.array([int(r["label"]) for r in train])
    cv = StratifiedKFold(3, shuffle=True, random_state=plan["seeds"]["inner_cv"])
    assignments = np.empty(len(train), dtype=int)
    cache = []
    for fold, (fit, held) in enumerate(cv.split(text, y)):
        assignments[held] = fold
        start = time.perf_counter()
        tfidf = vectorizer(original)
        x = tfidf.fit_transform([text[i] for i in fit])
        z = tfidf.transform([text[i] for i in held])
        sparse_seconds = time.perf_counter()-start
        start = time.perf_counter()
        svd = TruncatedSVD(n_components=128, n_iter=7, random_state=42)
        xd = svd.fit_transform(x)
        zd = svd.transform(z)
        dense_seconds = time.perf_counter()-start
        scaler = StandardScaler()
        xs, zs = scaler.fit_transform(xd), scaler.transform(zd)
        cache.append({"sparse": (x, z), "dense": (xd, zd), "scaled_dense": (xs, zs),
                      "y_fit": y[fit], "y_held": y[held],
                      "feature_seconds": {"sparse": sparse_seconds, "svd": dense_seconds},
                      "fit_ids_sha256": id_hash([train[i]["id"] for i in fit]),
                      "held_ids_sha256": id_hash([train[i]["id"] for i in held]),
                      "svd_explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum())})
        print(f"Prepared fold {fold}: train-only vocabulary, SVD and scaling", flush=True)
    write_csv(study/"inner_folds.csv", ["id", "inner_validation_fold"],
              ({"id": r["id"], "inner_validation_fold": int(assignments[i])} for i,r in enumerate(train)))
    return cache


def tune(train, plan, original, study):
    cache = prepare_folds(train, plan, original, study)
    results, observations, selected = [], [], {}
    for family in plan["families"]:
        initial = list(ParameterGrid(plan["initial_grids"][family]))
        seen, family_results = set(), []
        for stage in (1, 2):
            candidates = initial if stage == 1 else refinement(family, max(family_results, key=lambda r:r["mean_cv_auroc"])["parameters"])
            for params in candidates:
                key = signature(params)
                if key in seen:
                    continue
                seen.add(key)
                candidate_id = f"{family}_{len(seen):02d}"
                aucs, seconds, note_count = [], [], 0
                for fold, data in enumerate(cache):
                    x,z = data[representation(family, plan)]
                    model = classifier(family, params, plan["seeds"]["model"])
                    elapsed, notes = fit_record(model, x, data["y_fit"])
                    values = (model.decision_function(z) if family in ("lr", "lr_svd", "svm") else predict_ai(model,z))
                    from sklearn.metrics import roc_auc_score
                    auc = float(roc_auc_score(data["y_held"], values))
                    aucs.append(auc); seconds.append(elapsed); note_count += len(notes)
                    observations.append({"model": family, "candidate_id": candidate_id, "stage": stage,
                                         "parameters": params, "fold": fold, "auroc": auc,
                                         "classifier_fit_seconds": elapsed, "warnings": notes,
                                         "n_iter": int(model.n_iter_) if hasattr(model,"n_iter_") else None,
                                         "fit_ids_sha256": data["fit_ids_sha256"],
                                         "held_ids_sha256": data["held_ids_sha256"]})
                    save_json(study/"cv_progress.json", observations)
                record = {"model": family, "candidate_id": candidate_id, "stage": stage, "parameters": params,
                          "mean_cv_auroc": float(np.mean(aucs)), "std_cv_auroc": float(np.std(aucs, ddof=1)),
                          "fold_aurocs": aucs, "classifier_fit_seconds_sum": sum(seconds), "warning_count": note_count}
                family_results.append(record); results.append(record)
                save_json(study/"cv_results.json", results)
                print(f"CV {candidate_id} stage={stage} mean={record['mean_cv_auroc']:.6f} params={key}", flush=True)
        selected[family] = deepcopy(max(family_results, key=lambda r:r["mean_cv_auroc"])["parameters"])
        save_json(study/"chosen_parameters.json", selected)
    write_csv(study/"cv_results.csv", ["model","candidate_id","stage","parameters","mean_cv_auroc","std_cv_auroc","warning_count"],
              ({k:signature(r[k]) if k=="parameters" else r[k] for k in
                ("model","candidate_id","stage","parameters","mean_cv_auroc","std_cv_auroc","warning_count")} for r in results))
    save_json(study/"feature_fit_audit.json", [{k:v for k,v in d.items() if k not in
              ("sparse","dense","scaled_dense","y_fit","y_held")} for d in cache])
    return selected


def new_context():
    context = provenance()
    for name in ("expanded_q1.py", "q1_expanded_plan.json"):
        context["source_sha256"][f"experiments/supervised/{name}"] = sha256(HERE/name)
    return context


def run(prefix):
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in prefix):
        raise ValueError("Invalid prefix")
    plan, original = load_plans()
    base = ROOT/"results/supervised"
    study = base/f"{prefix}_study"
    names = [study]+[base/f"{prefix}_{f}" for f in plan["families"]]
    if any(p.exists() for p in names):
        raise FileExistsError("Use a NEW prefix; existing evidence is immutable")
    study.mkdir(parents=True)
    for name in ("expanded_q1.py","q1_expanded_plan.json","run_q1.py","q1_plan.json","requirements.txt"):
        shutil.copy2(HERE/name, study/name)
    context = new_context()
    save_json(study/"provenance.json", context)
    for name in ("train_ids.csv","validation_ids.csv","manifest.json"):
        shutil.copy2(SPLIT_DIR/name, study/name)
    train, val = load_partitions()
    start = time.perf_counter()
    chosen = tune(train,plan,original,study)
    tuning_seconds = time.perf_counter()-start
    train_x, val_x = [r["text"] for r in train], [r["text"] for r in val]
    y, vy = np.array([int(r["label"]) for r in train]),np.array([int(r["label"]) for r in val])
    comparison, predictions = [], {}
    for family in plan["families"]:
        folder = base/f"{prefix}_{family}"
        folder.mkdir()
        model = make_model(family, chosen[family],plan,original)
        save_json(folder/"config.json", {"run_id":folder.name,"study_id":study.name,"family":family,
                  "selected_parameters":chosen[family],"representation":representation(family,plan),
                  "plan":plan,"original_representation_plan":original,"provenance":context,
                  "split":json.loads((SPLIT_DIR/"manifest.json").read_text()),
                  "train_ids_file":f"../{study.name}/train_ids.csv",
                  "validation_ids_file":f"../{study.name}/validation_ids.csv",
                  "estimator_parameters":clean_params(model)})
        seconds, notes = fit_record(model,train_x,y)
        tp,vp = predict_ai(model,train_x),predict_ai(model,val_x)
        predictions[family] = vp
        fitted = model.calibrated_classifiers_[0].estimator if family=="svm" else model
        estimator = fitted.named_steps["classifier"]
        metrics = {"train":{"n":len(y),**score(y,tp)},"validation":{"n":len(vy),**score(vy,vp)},
                   "fit_seconds":seconds,"warnings":notes,"n_features":len(fitted.named_steps["tfidf"].vocabulary_),
                   "n_iter":int(estimator.n_iter_) if hasattr(estimator,"n_iter_") else None,
                   "svd_explained_variance_ratio_sum":float(fitted.named_steps["svd"].explained_variance_ratio_.sum()) if "svd" in fitted.named_steps else None,
                   "timing_definition":"single CPU thread; full feature extraction and calibration included; search excluded",
                   "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024),
                   "memory_note":"cumulative process high-water mark, not per-model memory",
                   "mlp_gradient_training_fraction":0.9 if family=="mlp" else None}
        save_json(folder/"metrics.json",metrics)
        write_csv(folder/"validation_predictions.csv",["id","label","p_ai"],
                  ({"id":r["id"],"label":r["label"],"p_ai":format(float(p),".17g")} for r,p in zip(val,vp,strict=True)))
        check(folder/"validation_predictions.csv","validation")
        joblib.dump(model,folder/"model.joblib")
        comparison.append({"model":family,"name":NAMES[family],"run_id":folder.name,
                           "representation":representation(family,plan),"parameters":signature(chosen[family]),
                           "train_auroc":metrics["train"]["auroc"],"validation_auroc":metrics["validation"]["auroc"],
                           "fit_seconds":seconds,"warning_count":len(notes)})
        write_csv(study/"comparison.csv",list(comparison[0]),comparison)
        print(f"HELD-OUT {family}: AUROC={metrics['validation']['auroc']:.6f} fit={seconds:.2f}s",flush=True)
    winner = max(comparison,key=lambda r:r["validation_auroc"])
    save_json(study/"selection.json",{"winner":winner["model"],"winner_run_id":winner["run_id"],
              "criterion":plan["selection"]["family_choice"],"scope":"Q1 candidate; team-wide choice open after Q3/Q5",
              "tuning_seconds_all_families":tuning_seconds,"total_wall_seconds_before_bootstrap":time.perf_counter()-start,
              "validation_previously_inspected":True,"test_or_leaderboard_used":False})
    save_json(study/"bootstrap.json",bootstrap(vy,predictions,winner["model"],plan["seeds"]["bootstrap"]))
    seed_results = []
    for seed in plan["mlp"]["secondary_seed_checks"]:
        model = make_model("mlp",chosen["mlp"],plan,original,seed)
        seconds, notes = fit_record(model,train_x,y)
        p = predict_ai(model,val_x)
        seed_results.append({"seed":seed,"selected_parameters":chosen["mlp"],
                             "validation":score(vy,p),"fit_seconds":seconds,"warnings":notes})
        write_csv(study/f"mlp_seed_{seed}_validation_predictions.csv",["id","label","p_ai"],
                  ({"id":r["id"],"label":r["label"],"p_ai":format(float(v),".17g")} for r,v in zip(val,p,strict=True)))
    save_json(study/"mlp_secondary_seeds.json",{"purpose":"stability only; not used for selection","runs":seed_results})
    print(f"FINISHED {study.name}; candidate={winner['run_id']}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix",required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.prefix)


if __name__=="__main__":
    main()
