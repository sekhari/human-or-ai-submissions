"""Confidence-filtered self-training; validation pseudo-labels exclude validation labels."""
import os,json,time,zipfile
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from config import OUT, DATA_ZIP
c=np.load(OUT/'enriched_features.npz');x=c['train'];xu=c['unlabeled'];xt=c['test']
with zipfile.ZipFile(DATA_ZIP) as z:
 train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'))
y=train.label.to_numpy();tr,va=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
comp=pd.read_csv(OUT/'tree_comparison.csv');cats=comp[comp.model.str.startswith('cat_rich')].sort_values('validation_auc',ascending=False);parent=cats.iloc[0].model
settings=json.loads((OUT/'tree_settings.json').read_text());params=settings[parent]['params']
vun=pd.read_csv(OUT/'tree_unlabeled_from_8k.csv');fun=pd.read_csv(OUT/'tree_unlabeled.csv')
# Predeclared two-model agreement; do not select confidence thresholds on hidden data.
names=['cat_rich4','cat_rich6']
def pseudo(d):
 p=d[names].to_numpy();pos=(p>.995).all(axis=1);neg=(p<.005).all(axis=1);mask=pos|neg
 return mask,pos[mask].astype(int)
vm,vy=pseudo(vun);fm,fy=pseudo(fun)
hard=pd.read_csv(OUT/'validation_protocol.csv').hard.to_numpy(dtype=bool)
rows=[];vpred={};tpred={}
for weight in [.2,.5]:
 start=time.perf_counter();m=CatBoostClassifier(**params)
 a=np.vstack([x[tr],xu[vm]]);lab=np.r_[y[tr],vy];weights=np.r_[np.ones(len(tr)),np.full(vm.sum(),weight)]
 m.fit(a,lab,sample_weight=weights);p=m.predict_proba(x[va])[:,1];name='pseudo_'+str(weight);vpred[name]=p
 row={'model':name,'parent':parent,'pseudo_rows_validation':int(vm.sum()),'pseudo_rows_full':int(fm.sum()),'weight':weight,'validation_auc':roc_auc_score(y[va],p),'testlike_auc':roc_auc_score(y[va][hard],p[hard]),'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True)
 full=CatBoostClassifier(**params);full.fit(np.vstack([x,xu[fm]]),np.r_[y,fy],sample_weight=np.r_[np.ones(len(y)),np.full(fm.sum(),weight)]);tpred[name]=full.predict_proba(xt)[:,1];full.save_model(str(OUT/(name+'.cbm')))
 pd.DataFrame(rows).to_csv(OUT/'pseudo_comparison.csv',index=False)
 pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':y[va],**vpred}).to_csv(OUT/'pseudo_validation.csv',index=False)
 pd.DataFrame({'id':test.id,**tpred}).to_csv(OUT/'pseudo_test.csv',index=False)
(OUT/'pseudo_config.json').write_text(json.dumps({'consensus_models':names,'threshold':.995,'parent':parent,'weights':[.2,.5],'validation_pseudo_counts':{'human':int((vy==0).sum()),'AI':int((vy==1).sum())},'full_pseudo_counts':{'human':int((fy==0).sum()),'AI':int((fy==1).sum())}},indent=2));print('PSEUDO COMPLETE',flush=True)
