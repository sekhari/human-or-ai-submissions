"""Nonlinear smooth decision boundaries over style features."""
import os
os.environ['OMP_NUM_THREADS']='3'
from pathlib import Path
import numpy as np,pandas as pd,json,time,zipfile
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.special import expit
from threadpoolctl import threadpool_limits
threadpool_limits(3)
from config import OUT, DATA_ZIP
c=np.load(OUT/'enriched_features.npz');x=c['train'][:,:99];xt=c['test'][:,:99]
with zipfile.ZipFile(DATA_ZIP) as z:train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'))
y=train.label.to_numpy();tr,va=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
s=StandardScaler().fit(x[tr]);a=np.clip(s.transform(x),-5,5);hard=pd.read_csv(OUT/'validation_protocol.csv').hard.to_numpy(dtype=bool)
rows=[];vp={};tp={}
for C in [1.,10.,100.]:
 start=time.perf_counter();m=SVC(C=C,gamma='scale',cache_size=1024,random_state=42);m.fit(a[tr],y[tr]);v=m.decision_function(a[va]);name='rbf_'+str(C);vp[name]=expit(v)
 row={'model':name,'C':C,'train_auc':roc_auc_score(y[tr],m.decision_function(a[tr])),'validation_auc':roc_auc_score(y[va],v),'testlike_auc':roc_auc_score(y[va][hard],v[hard]),'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True)
 sf=StandardScaler().fit(x);af=np.clip(sf.transform(x),-5,5);at=np.clip(sf.transform(xt),-5,5);mf=SVC(C=C,gamma='scale',cache_size=1024,random_state=42);mf.fit(af,y);tp[name]=expit(mf.decision_function(at))
 # Sigmoid margins are bounded ranking scores, not a calibrated probability claim.
 pd.DataFrame(rows).to_csv(OUT/'kernel_comparison.csv',index=False);pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':y[va],**vp}).to_csv(OUT/'kernel_validation.csv',index=False);pd.DataFrame({'id':test.id,**tp}).to_csv(OUT/'kernel_test.csv',index=False)
print('KERNEL COMPLETE',flush=True)
