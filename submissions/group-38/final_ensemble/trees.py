"""Controlled enriched-feature boosting comparison, with a fixed outer validation split."""
import os,json,time,sys,zipfile
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
threadpool_limits(4)
from config import OUT, DATA_ZIP
with zipfile.ZipFile(DATA_ZIP) as z:
 train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'));unlab=pd.read_csv(z.open('unlabeled.csv'))
c=np.load(OUT/'enriched_features.npz');x=c['train'];xt=c['test'];xu=c['unlabeled'];y=train.label.to_numpy()
tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=y,random_state=42)
# Domain similarity is an auxiliary classifier, not authorship labels or final features.
v=TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=3,sublinear_tf=True)
xd=v.fit_transform(pd.concat([train.text.iloc[tr],test.text]));domain=LogisticRegression(C=1,max_iter=1000,class_weight='balanced',random_state=42).fit(xd,np.r_[np.zeros(len(tr)),np.ones(len(test))])
d=domain.predict_proba(v.transform(train.text.iloc[va]))[:,1];hard=d>=np.quantile(d,.75)
pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':y[va],'test_likeness':d,'hard':hard}).to_csv(OUT/'validation_protocol.csv',index=False)
rows=[];vp={};tp={};up={};uvp={};settings={}
configs=[('cat_old',4,'old'),('cat_rich4',4,'all'),('cat_rich6',6,'all'),('xgb_rich3',3,'all'),('xgb_rich5',5,'all')]
# Resume completed configurations after cancelling the costly depth-8 pilot.
if (OUT/'tree_comparison.csv').exists():
 rows=pd.read_csv(OUT/'tree_comparison.csv').to_dict('records')
 settings=json.loads((OUT/'tree_settings.json').read_text())
 for fn,target in [('tree_validation.csv',vp),('tree_test.csv',tp),('tree_unlabeled.csv',up),('tree_unlabeled_from_8k.csv',uvp)]:
  frame=pd.read_csv(OUT/fn)
  for col in frame.columns:
   if col not in ['id','label']:target[col]=frame[col].to_numpy()
for name,depth,fs in configs:
 if name in settings:
  print('REUSE',name,flush=True)
  continue
 ix=np.arange(99) if fs=='old' else np.arange(x.shape[1]);a=x[:,ix];at=xt[:,ix];au=xu[:,ix];start=time.perf_counter()
 if name.startswith('cat'):
  p=dict(iterations=1800,depth=depth,learning_rate=.04,l2_leaf_reg=5,loss_function='Logloss',eval_metric='AUC',random_seed=42,thread_count=4,allow_writing_files=False,verbose=False)
  m=CatBoostClassifier(**p);m.fit(a[tr],y[tr],eval_set=(a[va],y[va]),early_stopping_rounds=150,verbose=False);rounds=m.tree_count_
 else:
  p=dict(n_estimators=1800,max_depth=depth,learning_rate=.035,min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=10,reg_alpha=.1,objective='binary:logistic',eval_metric='auc',tree_method='hist',n_jobs=4,random_state=42,early_stopping_rounds=150)
  m=XGBClassifier(**p);m.fit(a[tr],y[tr],eval_set=[(a[va],y[va])],verbose=False);rounds=m.best_iteration+1
 prob=m.predict_proba(a[va])[:,1];vp[name]=prob;uvp[name]=m.predict_proba(au)[:,1]
 row={'model':name,'features':len(ix),'rounds':rounds,'train_auc':roc_auc_score(y[tr],m.predict_proba(a[tr])[:,1]),'validation_auc':roc_auc_score(y[va],prob),'testlike_auc':roc_auc_score(y[va][hard],prob[hard]),'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True)
 # Refit fixed round count on all labeled data, without test labels.
 if name.startswith('cat'):p['iterations']=rounds;full=CatBoostClassifier(**p)
 else:p['n_estimators']=rounds;p.pop('early_stopping_rounds');full=XGBClassifier(**p)
 full.fit(a,y);tp[name]=full.predict_proba(at)[:,1];up[name]=full.predict_proba(au)[:,1]
 if name.startswith('cat'):full.save_model(str(OUT/(name+'.cbm')))
 else:full.save_model(OUT/(name+'.json'))
 settings[name]={'params':p,'feature_set':fs}
 pd.DataFrame(rows).to_csv(OUT/'tree_comparison.csv',index=False)
 pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':y[va],**vp}).to_csv(OUT/'tree_validation.csv',index=False)
 pd.DataFrame({'id':test.id,**tp}).to_csv(OUT/'tree_test.csv',index=False)
 pd.DataFrame({'id':unlab.id,**up}).to_csv(OUT/'tree_unlabeled.csv',index=False)
 pd.DataFrame({'id':unlab.id,**uvp}).to_csv(OUT/'tree_unlabeled_from_8k.csv',index=False)
 (OUT/'tree_settings.json').write_text(json.dumps(settings,indent=2))
print('TREES COMPLETE',flush=True)
