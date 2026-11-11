"""Fixed four-method recipe tested on topics excluded from supervised fitting.

Uses only the original 6400-row fitting partition. Calibration is drawn from the
remaining topics. Weights are locked; held-out topic labels never select them.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from src.calibration import RankCalibration
from src.robust_ensemble import RobustEnsemble
from src.encoder import DEVICE,predict as predict_legacy
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from src.calibration import ScoreCalibration
from src.encoder import train_encoder
from src.encoder_v2 import fit_encoder,predict_encoder
from src.robust_ensemble import short_lines
from train_robust import atomic_array

def legacy_scores(texts,directory):
    model=AutoModelForSequenceClassification.from_pretrained(directory,token=False,attn_implementation='eager').to(DEVICE)
    result=predict_legacy(model,AutoTokenizer.from_pretrained(directory,token=False),list(texts))
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def rank_features(values):
    return np.column_stack([np.rint(2*rankdata(v)).astype(np.int64) for v in values.T])


def control_score(values):
    return (rank_features(values)@np.array([55,25,20,0],dtype=np.int64))/(200*len(values))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--large-features',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(a.data/'train.csv',keep_default_na=False)
    t=tr.text.to_numpy();y=tr.label.to_numpy()
    digest=hashlib.sha256(tr[['id','text','label']].to_csv(index=False).encode()).hexdigest()
    cached=np.load(a.features)
    if str(cached['digest'])!=digest:raise ValueError('Feature cache dataset mismatch.')
    features=cached['features']
    recipe=json.loads(a.recipe.read_text());weights=np.asarray(recipe['weights']);mode=recipe['score_mode']
    if mode!='quantile' or recipe['lm_mode']!='both':raise ValueError('Use the frozen validated dual-LM quantile recipe.')
    if recipe['training_sha256']!=digest:raise ValueError('Recipe dataset mismatch.')
    candidate_features=features
    if recipe['lm_mode']!='small':
        if a.large_features is None:raise ValueError('This recipe requires --large-features.')
        large=np.load(a.large_features)
        if str(large['digest'])!=digest:raise ValueError('Larger LM dataset mismatch.')
        candidate_features=large['features'] if recipe['lm_mode']=='large' else np.c_[features,large['features'],large['features']-features]
    dev,_=train_test_split(np.arange(len(tr)),test_size=2000,stratify=y,random_state=3737)
    original_fit,_=train_test_split(dev,test_size=1600,stratify=y[dev],random_state=9407)
    vectorizer=TfidfVectorizer(stop_words='english',min_df=5,max_df=.4,max_features=40000,sublinear_tf=True)
    x=vectorizer.fit_transform(t[original_fit])
    z=normalize(TruncatedSVD(60,random_state=3737).fit_transform(x))
    groups=KMeans(6,n_init=5,random_state=3737).fit_predict(z)
    vocabulary=np.array(vectorizer.get_feature_names_out())
    top={g:vocabulary[np.argsort(np.asarray(x[groups==g].mean(0)).ravel())[-15:]].tolist() for g in range(6)}
    vocabulary_targets={'news':{'government','company','mr','uk','market','year','said'}}
    chosen={name:max(range(6),key=lambda g:len(set(top[g])&words)) for name,words in vocabulary_targets.items()}
    if len(set(chosen.values()))!=len(chosen):raise ValueError('Content groups could not be distinguished.')
    source=Path(__file__).parent
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+b''.join(path.read_bytes() for path in sorted((source/'src').glob('*.py')))).hexdigest()
    config={'dataset_sha256':digest,'source_sha256':fingerprint,'weights':weights.tolist(),'recipe':recipe,'topics':chosen,'topic_terms':top}
    config=json.loads(json.dumps(config))
    manifest=a.output/'config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Choose a fresh output directory for another experiment.')
    manifest.write_text(json.dumps(config,indent=2)+'\n')
    reports=[];all_labels=[];pooled_new={name:[] for name in ('original','flat','short_lines')};pooled_old={name:[] for name in pooled_new}
    for topic,group in chosen.items():
        held=original_fit[groups==group];rest=original_fit[groups!=group]
        fit,cal=train_test_split(rest,test_size=800,stratify=y[rest],random_state=9407)
        directory=a.output/topic;directory.mkdir(exist_ok=True)
        print(f'Content holdout {topic}: fit {len(fit)}, calibration {len(cal)}, held out {len(held)}',flush=True)
        np.savez(directory/'partitions.npz',fit=fit,calibration=cal,held=held)
        classical_path=directory/'classical.joblib'
        if classical_path.exists():models=joblib.load(classical_path)
        else:
            reference=RobustEnsemble.fit_classical(t[fit],y[fit],features[fit])
            lm=reference.surprisal
            candidate_lm=lm if recipe['lm_mode']=='small' else CatBoostClassifier(iterations=500,depth=4,learning_rate=.04,random_seed=3737,thread_count=5,verbose=False,allow_writing_files=False).fit(candidate_features[fit],y[fit])
            models={'reference':reference,'lm':lm,'candidate_lm':candidate_lm};joblib.dump(models,classical_path)
        reference=models['reference'];lm=models['lm']
        ordered=np.r_[cal,held];scores_path=directory/'encoder_scores.npz'
        if scores_path.exists():
            saved=np.load(scores_path);deberta=saved['deberta'];distil=saved['distil']
            if not np.array_equal(saved['ordered'],ordered):raise ValueError('Encoder score split mismatch.')
        else:
            deb_dir=directory/'deberta'
            if not (deb_dir/'training_complete.json').exists():fit_encoder(t[fit],y[fit],deb_dir)
            deberta=predict_encoder(t[ordered],deb_dir)
            distil_dir=directory/'distilroberta'
            if not (distil_dir/'training_complete.json').exists():
                model,tok=train_encoder(list(t[fit]),y[fit],log_every=100)
                model.save_pretrained(distil_dir);tok.save_pretrained(distil_dir)
                (distil_dir/'training_complete.json').write_text(json.dumps({'rows':len(fit),'epochs':2})+'\n')
                del model,tok;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
            distil=legacy_scores(t[ordered],distil_dir)
            atomic_array(scores_path,ordered=ordered,deberta=deberta,distil=distil)
        lm_score=lm.predict(features[ordered],prediction_type='RawFormulaVal')
        candidate_lm_score=models['candidate_lm'].predict(candidate_features[ordered],prediction_type='RawFormulaVal')
        raw={}
        for name,view in [('original',t[ordered]),('flat',[' '.join(s.split()) for s in t[ordered]]),
                          ('short_lines',[short_lines(s) for s in t[ordered]])]:
            raw[name]=np.c_[reference.parts(view,features[ordered])['reference'],candidate_lm_score,distil,deberta]
        calibration=RankCalibration().fit(np.concatenate([v[:len(cal)] for v in raw.values()]),np.tile(y[cal],len(raw)))
        report={'topic':topic,'terms':top[group],'fit_rows':len(fit),'calibration_rows':len(cal),
                'held_rows':len(held),'held_human':int(sum(y[held]==0)),'held_ai':int(sum(y[held]==1)),
                'candidate':{},'control':{}}
        arrays={'ids':tr.id.to_numpy()[held],'labels':y[held]}
        all_labels.append(y[held])
        for name,values in raw.items():
            test=values[len(cal):]
            new=calibration.transform(test)@weights
            control=test.copy();control[:,1]=lm_score[len(cal):]
            old=control_score(control)
            report['candidate'][name]=float(roc_auc_score(y[held],new));report['control'][name]=float(roc_auc_score(y[held],old))
            pooled_new[name].append(new);pooled_old[name].append(old)
            arrays['candidate_'+name]=new;arrays['control_'+name]=old
        atomic_array(directory/'scores.npz',**arrays)
        (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        print(json.dumps(report,indent=2),flush=True)
    labels=np.concatenate(all_labels)
    result={'weights_locked_before_testing':weights.tolist(),'score_mode':mode,'lm_mode':recipe['lm_mode'],'topics':reports,
            'pooled_candidate':{name:float(roc_auc_score(labels,np.concatenate(values))) for name,values in pooled_new.items()},
            'pooled_control':{name:float(roc_auc_score(labels,np.concatenate(values))) for name,values in pooled_old.items()},
            'scope':'Additional content group with frozen recipe; from the original fitting partition only, not a human-poetry benchmark.'}
    (a.output/'domain_ensemble.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
