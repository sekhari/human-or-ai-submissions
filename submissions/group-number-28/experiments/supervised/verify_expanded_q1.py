"""Audit the expanded search and all stored probabilities; optional fresh refits."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from human_ai.data import ROOT,load_partitions,read_csv,sha256
from human_ai.check_predictions import check
from .expanded_q1 import HERE,make_model,fit_record
from .run_q1 import predict_ai,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--refit",action="store_true")
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    base=ROOT/"results/supervised"
    study=base/f"{args.prefix}_study"
    plan=json.loads((study/"q1_expanded_plan.json").read_text())
    original=json.loads((study/"q1_plan.json").read_text())
    context=json.loads((study/"provenance.json").read_text())
    for name in ("expanded_q1.py","q1_expanded_plan.json","run_q1.py","q1_plan.json","requirements.txt"):
        expected=context["source_sha256"][f"experiments/supervised/{name}"]
        assert sha256(study/name)==expected,name
        assert sha256(HERE/name)==expected,name
    train,val=load_partitions()
    x,z=[r["text"] for r in train],[r["text"] for r in val]
    y,vy=np.array([int(r["label"]) for r in train]),np.array([int(r["label"]) for r in val])
    folds=read_csv(study/"inner_folds.csv")
    assert [r["id"] for r in folds]==[r["id"] for r in train]
    for fold,(_,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=142).split(x,y)):
        assert all(int(folds[i]["inner_validation_fold"])==fold for i in held)
    cv=json.loads((study/"cv_results.json").read_text())
    progress=json.loads((study/"cv_progress.json").read_text())
    chosen=json.loads((study/"chosen_parameters.json").read_text())
    summary=read_csv(study/"comparison.csv")
    audit={"source_hashes_match":True,"folds_match_training_only_split":True,"n_candidates":len(cv),
           "n_candidate_fold_fits":len(progress),"models":{}}
    with threadpool_limits(limits=1):
        for family in plan["families"]:
            candidates=[r for r in cv if r["model"]==family]
            best=max(candidates,key=lambda r:r["mean_cv_auroc"])
            assert best["parameters"]==chosen[family]
            for r in candidates:
                foldscores=[v["auroc"] for v in progress if v["candidate_id"]==r["candidate_id"]]
                assert len(foldscores)==3
                assert abs(float(np.mean(foldscores))-r["mean_cv_auroc"])<1e-14
            folder=base/f"{args.prefix}_{family}"
            assert check(folder/"validation_predictions.csv","validation")==2000
            rows=read_csv(folder/"validation_predictions.csv")
            assert [r["id"] for r in rows]==[r["id"] for r in val]
            recorded=np.array([float(r["p_ai"]) for r in rows])
            model=joblib.load(folder/"model.joblib")
            restored=predict_ai(model,z)
            np.testing.assert_allclose(restored,recorded,atol=1e-14,rtol=0)
            auc=float(roc_auc_score(vy,recorded))
            metrics=json.loads((folder/"metrics.json").read_text())
            assert abs(metrics["validation"]["auroc"]-auc)<1e-14
            assert abs(float(next(r for r in summary if r["model"]==family)["validation_auroc"])-auc)<1e-14
            result={"n_candidates":len(candidates),"n_validation":2000,"validation_auroc":auc,
                    "model_reload_max_probability_error":float(np.max(abs(restored-recorded))),
                    "best_setting_matches_training_cv":True}
            if args.refit:
                fresh=make_model(family,chosen[family],plan,original)
                seconds,notes=fit_record(fresh,x,y)
                predicted=predict_ai(fresh,z)
                error=float(np.max(abs(predicted-recorded)))
                np.testing.assert_allclose(predicted,recorded,atol=1e-10,rtol=0)
                result.update({"fresh_refit_max_probability_error":error,"fresh_refit_seconds":seconds,"warnings":notes})
            audit["models"][family]=result
            print(f"Verified {family}: {auc:.6f}",flush=True)
    audit["all_checks_passed"]=True
    out.parent.mkdir(parents=True,exist_ok=True)
    save_json(out,audit)
    print(f"Saved audit {out}")


if __name__=="__main__":main()
