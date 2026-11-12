"""Run from the Kaggle project using the SVM Python environment plus wordfreq 3.1.1."""
from pathlib import Path
import ast,json,argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=HERE.parent/'data')
parser.add_argument('--split',type=Path,default=HERE.parent/'jianing_svm/data/splits/seed42_stratified_8000_2000.csv')
parser.add_argument('--reference',type=Path,default=HERE.parent/'abishek_ensemble/rank_average.ipynb')
parser.add_argument('--submission',action='store_true')
args=parser.parse_args()
DATA=args.data
out=HERE/'results';out.mkdir(exist_ok=True)
# Reuse the definitions from Abishek's notebook without running its search or export cells.
scope={}
for cell in json.loads(args.reference.read_text())['cells']:
 if cell['cell_type']!='code': continue
 nodes=[]
 for node in ast.parse(''.join(cell['source'])).body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)):nodes.append(node)
  if isinstance(node,ast.Assign) and any(getattr(t,'id','') in {'SENT_SPLIT','WORD_RE','TOK_RE','FUNCTION_WORDS','FIRST_PERSON','SECOND_PERSON','HEDGES','AI_CONNECTIVES'} for t in node.targets):nodes.append(node)
 exec(compile(ast.Module(nodes,[]),'abi_definitions','exec'),scope)
df=pd.read_csv(DATA/'train.csv');u=pd.read_csv(DATA/'unlabeled.csv');test=pd.read_csv(DATA/'test.csv')
assert len(df)==10000 and len(u)==3000 and set(u.columns)=={'id','text'}
canonical = lambda values: set(values.str.lower().str.split().str.join(' '))
assert u.text.notna().all() and len(canonical(u.text))==3000
assert canonical(u.text).isdisjoint(canonical(df.text))
a,b=train_test_split(df,test_size=2000,stratify=df.label,random_state=42)
ref=pd.read_csv(args.split)
assert set(b.id)==set(ref.loc[ref.partition=='validation','id'])
# Refit binary vocabularies on the current labeled fold plus unlabeled passages.
# NB counts, logistic regression, boosting, and stacking use true training labels only.
def fit_features(factory, labeled):
 vectorizer = scope[factory]()
 vectorizer.fit(pd.concat([labeled, u.text], ignore_index=True))
 return vectorizer, vectorizer.transform(labeled)

def stack(train,evaluation):
 folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(train,train.label))
 oof=np.zeros((len(train),5));train_probs=np.zeros_like(oof);ev=np.zeros((len(evaluation),5))
 for j,(fn,C) in enumerate([('word_vectorizer',.3),('char_vectorizer',.03),('skeleton_vectorizer',.1),('charx_vectorizer',.1)]):
  for inside,outside in folds:
   v,x=fit_features(fn,train.iloc[inside].text)
   m=scope['NBLogReg'](C=C).fit(x,train.label.iloc[inside])
   oof[outside,j]=m.predict_proba(v.transform(train.iloc[outside].text))[:,1]
  v,x=fit_features(fn,train.text);m=scope['NBLogReg'](C=C).fit(x,train.label)
  train_probs[:,j]=m.predict_proba(x)[:,1];ev[:,j]=m.predict_proba(v.transform(evaluation.text))[:,1]
  print(len(train),fn,flush=True)
 x=scope['stylo_frame'](train.text);z=scope['stylo_frame'](evaluation.text)
 def hgb():return HistGradientBoostingClassifier(max_iter=400,max_leaf_nodes=31,learning_rate=.1,random_state=42)
 for inside,outside in folds:
  m=hgb().fit(x.iloc[inside],train.label.iloc[inside]);oof[outside,4]=m.predict_proba(x.iloc[outside])[:,1]
 m=hgb().fit(x,train.label);train_probs[:,4]=m.predict_proba(x)[:,1];ev[:,4]=m.predict_proba(z)[:,1]
 meta=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(oof,train.label)
 return meta.predict_proba(train_probs)[:,1],meta.predict_proba(ev)[:,1]
with threadpool_limits(limits=2):
 if args.submission:
  _,pred=stack(df,test)
  sample=pd.read_csv(DATA/'sample_submission.csv')
  keyed=pd.Series(pred,index=test.id)
  assert keyed.index.is_unique and set(keyed.index)==set(sample.id) and np.isfinite(pred).all()
  sample['label']=keyed.loc[sample.id].to_numpy()
  assert len(sample)==3000 and sample.label.between(0,1).all()
  sample.to_csv(out/'stack_unlabeled.csv',index=False)
  raise SystemExit(0)
 tr,va=stack(a,b)
 vals=b.assign(prob=va,words=b.text.str.split().str.len()); vals.drop(columns='text').to_csv(out/'validation.csv',index=False)
 print('AUC',roc_auc_score(a.label,tr),roc_auc_score(b.label,va),flush=True)
 pd.DataFrame({'id':a.id,'label':a.label,'prob':tr}).to_csv(out/'train.csv',index=False)
