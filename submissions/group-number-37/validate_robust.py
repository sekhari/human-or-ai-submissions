"""Reproduce the fixed global audit; run train_robust.py --validation first."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from src.lm_features import MODELS,extract,combine
from src.robust_ensemble import short_lines
from src.encoder_v2 import predict_encoder
from predict_robust import DEVICE,legacy_predict,AutoModelForSequenceClassification,AutoTokenizer
from train_robust import partitions,atomic_array

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    a=p.parse_args();bundle=joblib.load(a.model/'model.joblib')
    if bundle['stage']!='validation':raise ValueError('Audit requires a validation fit, not a final fit that has seen audit labels.')
    frame=pd.read_csv(a.data/'train.csv',keep_default_na=False);texts=frame.text.to_numpy();y=frame.label.to_numpy()
    fit,_,_,audit=partitions(y)
    matrices=[]
    for model_id,name in zip(MODELS,('small','large')):
        path=a.model/f'{name}_features.npz'
        matrices.append(np.load(path)['features'] if path.exists() else extract(texts,model_id))
    features=combine(*matrices)
    small=CatBoostClassifier(iterations=500,depth=4,learning_rate=.04,random_seed=3737,thread_count=5,
                            verbose=False,allow_writing_files=False).fit(matrices[0][fit],y[fit])
    lm_control=small.predict(matrices[0][audit],prediction_type='RawFormulaVal')
    deb=predict_encoder(texts[audit],a.model/bundle['deberta_dir'])
    distil_dir=a.model/bundle['distil_dir']
    model=AutoModelForSequenceClassification.from_pretrained(distil_dir,token=False,attn_implementation='eager').to(DEVICE)
    distil=legacy_predict(model,AutoTokenizer.from_pretrained(distil_dir,token=False),list(texts[audit]))
    report={};arrays={'ids':frame.id.to_numpy()[audit],'labels':y[audit]}
    for name,transform in [('original',lambda s:s),('flat',lambda s:' '.join(s.split())),('short_lines',short_lines)]:
        view=[transform(s) for s in texts[audit]]
        raw=bundle['ensemble'].components(view,features[audit],distil,deb)
        candidate=bundle['ensemble'].score(view,features[audit],distil,deb)
        # Canonical committed three-way blend, on exactly the same fit rows.
        control=sum(w*np.rint(2*rankdata(column,method='average')).astype(np.int64) for w,column in zip((55,25,20),(raw[:,0],lm_control,distil)))/(200*len(audit))
        report[name]={'candidate_auc':roc_auc_score(y[audit],candidate),'control_auc':roc_auc_score(y[audit],control)}
        arrays['candidate_'+name]=candidate;arrays['control_'+name]=control
    atomic_array(a.model/'audit_scores.npz',**arrays)
    (a.model/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
