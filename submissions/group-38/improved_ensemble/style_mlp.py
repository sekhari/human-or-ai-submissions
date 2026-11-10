"""Lightweight neural network over explicit style features, with training-only scaling."""
import json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import joblib
from style_features import features,train,test,OUT
threadpool_limits(2)
x=np.array([features(t) for t in train.text],dtype=np.float32);xt=np.array([features(t) for t in test.text],dtype=np.float32)
np.savez(OUT/'style_features_cache.npz',train=x,test=xt)
tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.label,random_state=42)
# The scaler excludes both the team's outer validation set and the internal early-stopping set.
inner_fit,inner_stop=train_test_split(tr,test_size=.1,stratify=train.label.iloc[tr],random_state=42)
scaler=StandardScaler().fit(x[inner_fit]);z=scaler.transform(x);zt=scaler.transform(xt)
rows=[];best=None
# Explicit inner stopping by AUROC; no outer validation labels enter gradient updates.
for hidden,alpha in [((64,32),.1),((64,32),1.),((128,64),.1),((128,64),1.)]:
 m=MLPClassifier(hidden_layer_sizes=hidden,activation='relu',alpha=alpha,batch_size=128,learning_rate_init=.001,max_iter=1,early_stopping=False,random_state=42)
 bestinner=-1;patience=0;state=None;epoch=0;start=time.perf_counter()
 for epoch in range(1,151):
  m.partial_fit(z[inner_fit],train.label.iloc[inner_fit],classes=np.array([0,1]))
  ia=roc_auc_score(train.label.iloc[inner_stop],m.predict_proba(z[inner_stop])[:,1])
  if ia>bestinner+1e-5:
   bestinner=ia;patience=0;state=joblib.dumps(m) if hasattr(joblib,'dumps') else None
   import copy
   selected=copy.deepcopy(m);chosen_epoch=epoch
  else:patience+=1
  if patience>=15:break
 p=selected.predict_proba(z[va])[:,1]
 row={'hidden':list(hidden),'alpha':alpha,'epochs':chosen_epoch,'stopped_epoch':epoch,'inner_auroc':bestinner,'validation_auc':roc_auc_score(train.label.iloc[va],p),'train_auc':roc_auc_score(train.label.iloc[tr],selected.predict_proba(z[tr])[:,1]),'seconds':time.perf_counter()-start}
 rows.append(row);print(row,flush=True)
 if best is None or row['validation_auc']>best['validation_auc']:best=row;bp=p
 pd.DataFrame(rows).to_csv(OUT/'mlp_results.csv',index=False)
pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':train.label.iloc[va].to_numpy(),'mlp':bp}).to_csv(OUT/'mlp_validation.csv',index=False)
scaler=StandardScaler().fit(x);z=scaler.transform(x);zt=scaler.transform(xt)
# Two predeclared seeds reduce optimization noise. Epoch count fixed from inner early stopping.
parts=[]
for seed in [42,43]:
 m=MLPClassifier(hidden_layer_sizes=tuple(best['hidden']),activation='relu',alpha=best['alpha'],batch_size=128,learning_rate_init=.001,max_iter=1,early_stopping=False,random_state=seed)
 for epoch in range(best['epochs']):m.partial_fit(z,train.label,classes=np.array([0,1]))
 parts.append(m.predict_proba(zt)[:,1]);joblib.dump({'scaler':scaler,'model':m},OUT/f'style_mlp_full_{seed}.joblib')
pd.DataFrame({'id':test.id,'mlp':np.mean(parts,axis=0)}).to_csv(OUT/'mlp_test.csv',index=False)
(OUT/'mlp_summary.json').write_text(json.dumps(best,indent=2));print('MLP COMPLETE',flush=True)
