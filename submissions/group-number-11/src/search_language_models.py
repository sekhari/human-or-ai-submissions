"""Language features plus style, with native and AI-only format stress tests.

Frozen language features are corpus-independent. If they retain native format,
synthetic style-view diagnostics are marked incomplete until the language
features have also been recomputed for transformed validation passages.
"""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from advanced_features import load_lm_matrix
from search_optimization import get_features,model_for
from train import ROOT,SEED


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--lm-dirs',nargs='+',default=['lm']);ap.add_argument('--name',default='language_small')
    ap.add_argument('--no-lm',action='store_true');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization';df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy()
    tr,va=train_test_split(np.arange(len(df)),test_size=2000,stratify=y,random_state=SEED)
    features=get_features(df,art/'rich_train.joblib')
    lm=None if args.no_lm else load_lm_matrix(art,'train',args.lm_dirs)
    invariant=args.no_lm or all(json.loads((art/d/'lm_metadata.json').read_text())['normalization']=='canonical whitespace' for d in args.lm_dirs)
    configurations=[]
    for view in ['native','raw_flat','canonical']:
        for leaves in [7,15,31]:configurations.append({'view':view,'kind':'lgb','leaves':leaves})
        for depth in [5]:configurations.append({'view':view,'kind':'cat','depth':depth})
    results=[];models={};predictions={};variants=['raw','flat','single','double']
    for config in configurations:
        if lm is not None:config.update(lm=True,lm_dirs=args.lm_dirs)
        view=config['view'];source={v:features['flat' if view=='canonical' else v] for v in variants}
        matrices={v:np.column_stack([x,lm]) if lm is not None else x for v,x in source.items()}
        views=['raw','flat'] if view=='raw_flat' else ['flat' if view=='canonical' else 'raw']
        xt=np.concatenate([matrices[v][tr] for v in views]);yt=np.tile(y[tr],len(views))
        model=model_for(config);start=time.perf_counter();model.fit(xt,yt)
        ps={v:model.predict_proba(matrices[v][va])[:,1] for v in variants}
        aucs={v:roc_auc_score(y[va],p) for v,p in ps.items()}
        ai_stress={v:roc_auc_score(y[va],np.where(y[va]==1,ps[v],ps['raw'])) for v in variants[1:]}
        name=f"{view}_{config['kind']}_{config.get('leaves',config.get('depth'))}"
        row={'name':name,'config':config,'aucs':aucs,'AI_only_aucs':ai_stress,
             'native_train_auc':roc_auc_score(y[tr],model.predict_proba(matrices['raw'][tr])[:,1]),
             'seconds':time.perf_counter()-start,'format_stress_verified':invariant}
        print(json.dumps(row),flush=True);results.append(row);models[name]=model;predictions[name]=ps
        (out/(args.name+'.json')).write_text(json.dumps({'records':results,'lm_dirs':args.lm_dirs if not args.no_lm else [],
            'stress_caveat':None if invariant else 'LM features retained native format for these style-view diagnostics. Do not treat them as full-pipeline format invariance.'},indent=2))
    joblib.dump({'records':results,'models':models,'predictions':predictions,'tr':tr,'va':va},art/(args.name+'.joblib'),compress=3)


if __name__=='__main__':main()
