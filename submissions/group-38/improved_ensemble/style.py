"""Topic-light observable style features: a complementary generalization experiment."""
import re,zipfile,json,collections,time,os
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
from config import OUT, DATA_ZIP
with zipfile.ZipFile(DATA_ZIP) as z:
 train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'))
WORDS=['the','a','an','and','or','but','because','however','therefore','thus','moreover','furthermore','in','of','to','for','with','as','that','which','this','these','it','its','i','my','we','our','you','your','they','their','is','are','was','were','be','been','have','has','had','would','could','should','can','may','might','not','no','very','also','such','than','then','so','if','while','although','overall','important','significant','various','ensure','understanding','conclusion']
def features(t):
 w=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",t.lower());n=max(len(w),1);ch=max(len(t),1);c=collections.Counter(w);lens=np.array([len(v) for v in w] or [0]);sent=[re.findall(r'\b\w+\b',s) for s in re.split(r'[.!?]+',t) if s.strip()];sl=np.array([len(s) for s in sent] or [0]);freq=np.array(list(c.values()) or [1])/n
 f=[np.log1p(n),np.log1p(ch),len(c)/n,sum(v==1 for v in c.values())/n,float(lens.mean()),float(lens.std()),np.mean(lens>=7),np.mean(lens<=3),float(sl.mean()),float(sl.std()),float(sl.max()),len(sent)/n,t.count('\n')/ch,len(re.findall(r'\n\s*\n',t))/ch,sum(x.isupper() for x in t)/ch,sum(x.isdigit() for x in t)/ch,sum(x.isspace() for x in t)/ch,-float(np.sum(freq*np.log(freq+1e-12))),sum(v*v for v in c.values())/n**2]
 f += [t.count(x)/ch for x in ['.',',',';',':','!','?','-','—','(',')','"',"'",'\u2019','\u201c']]
 f += [c[x]/n for x in WORDS]
 f += [len(re.findall(r"\b\w+(?:n't|'re|'ve|'ll|'d|'m)\b",t.lower()))/n]
 return f
x=np.array([features(t) for t in train.text],dtype=np.float32);xt=np.array([features(t) for t in test.text],dtype=np.float32)
tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.label,random_state=42)
rows=[];best=None
for leaves,l2 in [(7,10),(15,10),(15,1)]:
 start=time.perf_counter();m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=leaves,l2_regularization=l2,early_stopping=True,random_state=42);m.fit(x[tr],train.label.iloc[tr]);p=m.predict_proba(x[va])[:,1]
 row={'leaves':leaves,'l2':l2,'train_auc':roc_auc_score(train.label.iloc[tr],m.predict_proba(x[tr])[:,1]),'validation_auc':roc_auc_score(train.label.iloc[va],p),'seconds':time.perf_counter()-start};rows.append(row);print(row,flush=True)
 if best is None or row['validation_auc']>best['validation_auc']:best=row;bp=p
pd.DataFrame(rows).to_csv(OUT/'style_results.csv',index=False)
pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':train.label.iloc[va].to_numpy(),'style':bp}).to_csv(OUT/'style_validation.csv',index=False)
m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=best['leaves'],l2_regularization=best['l2'],early_stopping=True,random_state=42);m.fit(x,train.label)
pd.DataFrame({'id':test.id,'style':m.predict_proba(xt)[:,1]}).to_csv(OUT/'style_test.csv',index=False)
print('STYLE COMPLETE',flush=True)
