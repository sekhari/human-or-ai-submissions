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
