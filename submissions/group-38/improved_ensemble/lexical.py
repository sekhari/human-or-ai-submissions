import os
os.environ['OMP_NUM_THREADS']='2'
import zipfile,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
threadpool_limits(2)
from config import OUT, DATA_ZIP
with zipfile.ZipFile(DATA_ZIP) as z:
 train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'))
tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.label,random_state=42)
y=train.label.to_numpy(); rows=[]; preds={}; best={}
def vecs():
 return [TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=120000,sublinear_tf=True,dtype=np.float32),TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=80000,sublinear_tf=True,dtype=np.float32)]
def ratio(x,y):
 p=1+np.asarray(x[y==1].sum(axis=0)).ravel();n=1+np.asarray(x[y==0].sum(axis=0)).ravel()
 return np.log((p/p.sum())/(n/n.sum())).astype(np.float32)
vs=vecs();a=[];b=[]
for v in vs:
 a.append(v.fit_transform(train.text.iloc[tr]));b.append(v.transform(train.text.iloc[va]));print('FEATURES',a[-1].shape,flush=True)
for kind in ['word','combined','nb_combined']:
 x=a[0] if kind=='word' else hstack(a,format='csr')
 xv=b[0] if kind=='word' else hstack(b,format='csr')
 r=ratio(x,y[tr]) if kind.startswith('nb') else np.ones(x.shape[1],dtype=np.float32)
 x=x.multiply(r).tocsr();xv=xv.multiply(r).tocsr()
 for c in [.25,1.,4.]:
  start=time.perf_counter();m=LogisticRegression(C=c,solver='liblinear',max_iter=1000,random_state=42);m.fit(x,y[tr]);p=m.predict_proba(xv)[:,1]
  row={'model':kind,'C':c,'train_auc':roc_auc_score(y[tr],m.predict_proba(x)[:,1]),'validation_auc':roc_auc_score(y[va],p),'seconds':time.perf_counter()-start};rows.append(row);print(row,flush=True)
  if kind not in best or row['validation_auc']>best[kind]['validation_auc']:
   best[kind]=row;preds[kind]=p
 pd.DataFrame(rows).to_csv(OUT/'lexical_tuning.csv',index=False)
pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':y[va],**preds}).to_csv(OUT/'lexical_validation.csv',index=False)
# Refit each selected model on all supplied labeled examples.
fullvs=vecs();af=[];bf=[]
for v in fullvs:
 af.append(v.fit_transform(train.text));bf.append(v.transform(test.text))
fullpred={}
for kind,row in best.items():
 x=af[0] if kind=='word' else hstack(af,format='csr');xt=bf[0] if kind=='word' else hstack(bf,format='csr')
 r=ratio(x,y) if kind.startswith('nb') else np.ones(x.shape[1],dtype=np.float32)
 m=LogisticRegression(C=row['C'],solver='liblinear',max_iter=1000,random_state=42);m.fit(x.multiply(r).tocsr(),y);fullpred[kind]=m.predict_proba(xt.multiply(r).tocsr())[:,1]
pd.DataFrame({'id':test.id,**fullpred}).to_csv(OUT/'lexical_test.csv',index=False)
(OUT/'lexical_best.json').write_text(json.dumps(best,indent=2))
print('LEXICAL COMPLETE',flush=True)
