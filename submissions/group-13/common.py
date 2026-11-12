"""Portable input, provenance and output checks; human=0, AI=1."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent
def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_data(path,labeled=False):
    d=pd.read_csv(path);required=['id','text']+(['label'] if labeled else [])
    if any(k not in d for k in required) or not d.id.is_unique or d[required].isna().any().any():raise ValueError('Invalid schema, duplicate IDs, or missing values')
    if not d.text.map(lambda t:isinstance(t,str) and bool(t.strip())).all():raise ValueError('Text must be nonempty strings')
    if labeled and set(d.label.unique())!={0,1}:raise ValueError('Training labels must contain both 0 and 1')
    return d
def split_data(d,seed=42):
    a,b=train_test_split(np.arange(len(d)),test_size=.2,random_state=seed,stratify=d.label)
    return d.iloc[a].copy(),d.iloc[b].copy()
def metrics(y,p):return dict(AUROC=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))
def write_predictions(path,d,p):
    p=np.asarray(p,dtype=float)
    if len(p)!=len(d) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid probabilities')
    path=Path(path)
    if path.exists():raise FileExistsError(f'Refusing to replace {path}; choose a new --out')
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':d.id.to_numpy(),'label':p}).to_csv(path,index=False)
    check=pd.read_csv(path)
    assert check.id.is_unique and np.array_equal(check.id,d.id) and np.allclose(check.label,p,rtol=0,atol=1e-15)
