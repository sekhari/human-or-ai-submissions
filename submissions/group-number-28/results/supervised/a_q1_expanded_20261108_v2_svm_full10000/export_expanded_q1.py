"""Refit the expanded-study winner on supplied labels and export a local CSV."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from human_ai.check_predictions import check
from human_ai.data import ROOT,load_partitions,raw_path,read_csv,sha256,write_csv
from .expanded_q1 import HERE,make_model,fit_record,new_context
from .run_q1 import predict_ai,save_json,score,clean_params


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    base=ROOT/"results/supervised"
    study=base/f"{args.prefix}_study"
    selection=json.loads((study/"selection.json").read_text())
    source=base/selection["winner_run_id"]
    source_config=json.loads((source/"config.json").read_text())
    context=json.loads((study/"provenance.json").read_text())
    for name in ("expanded_q1.py","q1_expanded_plan.json","run_q1.py","q1_plan.json","requirements.txt"):
        expected=context["source_sha256"][f"experiments/supervised/{name}"]
        if sha256(HERE/name)!=expected:raise ValueError(f"Changed execution source: {name}")
    output=Path(args.output).resolve()
    receipt=output.with_suffix(".metadata.json")
    folder=base/f"{source.name}_full10000"
    if any(p.exists() for p in (output,receipt,folder)):raise FileExistsError("Use fresh artifact/output paths")
    plan=source_config["plan"]
    original=source_config["original_representation_plan"]
    family=source_config["family"]
    params=source_config["selected_parameters"]
    train,val=load_partitions()
    records=train+val
    y=np.array([int(r["label"]) for r in records])
    text=[r["text"] for r in records]
    test=read_csv(raw_path("test.csv"))
    model=make_model(family,params,plan,original)
    folder.mkdir()
    config={"run_id":folder.name,"source_run_id":source.name,"study_id":study.name,
            "family":family,"selected_parameters":params,"n_train":len(records),
            "training_order":"canonical training IDs followed by canonical validation IDs",
            "held_out_evaluation":None,"scope":"expanded Q1 candidate; team final selection remains open",
            "plan":plan,"original_representation_plan":original,"provenance":new_context(),
            "estimator_parameters":clean_params(model),"export_source_sha256":sha256(Path(__file__)),
            "test_sha256":sha256(raw_path("test.csv"))}
    save_json(folder/"config.json",config)
    write_csv(folder/"train_ids.csv",["id"],({"id":r["id"]} for r in records))
    shutil.copy2(__file__,folder/"export_expanded_q1.py")
    with threadpool_limits(limits=1):
        seconds,notes=fit_record(model,text,y)
        tp=predict_ai(model,text)
        p=predict_ai(model,[r["text"] for r in test])
        joblib.dump(model,folder/"model.joblib")
        restored=joblib.load(folder/"model.joblib")
        np.testing.assert_array_equal(p,predict_ai(restored,[r["text"] for r in test]))
    save_json(folder/"metrics.json",{"train":{"n":len(records),**score(y,tp)},"validation":None,
              "fit_seconds":seconds,"warnings":notes,"note":"no held-out estimate after full-data refit"})
    output.parent.mkdir(parents=True,exist_ok=True)
    write_csv(output,["id","label"],({"id":r["id"],"label":format(float(v),".17g")} for r,v in zip(test,p,strict=True)))
    count=check(output,"submission")
    save_json(receipt,{"model_run_id":folder.name,"source_run_id":source.name,"n_test":count,
              "csv_sha256":sha256(output),"full_refit":True,"uploaded_to_kaggle":False,
              "model_reload_prediction_max_error":0.0,"created_at_utc":datetime.now(timezone.utc).isoformat()})
    print(f"Validated {count}-row local candidate: {output}; no Kaggle upload")


if __name__=="__main__":main()
