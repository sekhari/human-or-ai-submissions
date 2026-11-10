import json,zipfile,hashlib,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from style_features import train,test,OUT
from config import DATA_ZIP
threadpool_limits(2)
DEST=OUT/'deliverables';DEST.mkdir(parents=True,exist_ok=True)
c=np.load(OUT/'style_features_cache.npz');x=c['train']
tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.label,random_state=42)
best=json.loads((OUT/'mlp_summary.json').read_text());scaler=StandardScaler().fit(x[tr]);z=scaler.transform(x);vp=[]
for seed in [42,43]:
 m=MLPClassifier(hidden_layer_sizes=tuple(best['hidden']),activation='relu',alpha=best['alpha'],batch_size=128,learning_rate_init=.001,max_iter=1,early_stopping=False,random_state=seed)
 for epoch in range(best['epochs']):m.partial_fit(z[tr],train.label.iloc[tr],classes=np.array([0,1]))
 vp.append(m.predict_proba(z[va])[:,1])
v=pd.read_csv(OUT/'style_validation.csv')
expected=train.id.iloc[va].to_numpy();assert np.array_equal(v.id,expected)
v['mlp']=np.mean(vp,axis=0)
l=pd.read_csv(OUT/'lexical_validation.csv');assert np.array_equal(v.id,l.id)
v['combined']=l.combined
candidates={'style_only':v['style'].to_numpy(),'style75_mlp25':.75*v['style']+.25*v.mlp,'style50_mlp50':.5*v['style']+.5*v.mlp,'style75_mlp15_lexical10':.75*v['style']+.15*v.mlp+.10*v.combined,'mlp_only':v.mlp.to_numpy()}
word_count=train.text.iloc[va].str.split().str.len().to_numpy()
rows=[]
for name,p in candidates.items():
 row={'model':name,'validation_auc':roc_auc_score(v.label,p)}
 for lo,hi in [(0,150),(150,250),(250,10000)]:
  mask=(word_count>=lo)&(word_count<hi)
  row[f'length_{lo}_{hi}_auc']=roc_auc_score(v.label[mask],np.asarray(p)[mask])
 rows.append(row)
results=pd.DataFrame(rows).sort_values('validation_auc',ascending=False);print(results.to_string(index=False),flush=True)
chosen=results.iloc[0]['model'];v['selected_probability_ai']=candidates[chosen];v.to_csv(OUT/'improved_validation.csv',index=False)
a=pd.read_csv(OUT/'style_test.csv');b=pd.read_csv(OUT/'mlp_test.csv');d=pd.read_csv(OUT/'lexical_test.csv')
assert np.array_equal(a.id,b.id) and np.array_equal(a.id,d.id) and np.array_equal(a.id,test.id)
tc={'style_only':a['style'].to_numpy(),'style75_mlp25':.75*a['style']+.25*b.mlp,'style50_mlp50':.5*a['style']+.5*b.mlp,'style75_mlp15_lexical10':.75*a['style']+.15*b.mlp+.1*d.combined,'mlp_only':b.mlp.to_numpy()}
with zipfile.ZipFile(DATA_ZIP) as archive:sample=pd.read_csv(archive.open('sample_submission.csv'))
def save(name,p):
 pred=pd.DataFrame({'id':a.id,'label':p})
 assert np.isfinite(pred.label).all() and pred.label.between(0,1).all()
 sub=sample[['id']].merge(pred,on='id',how='left',sort=False,validate='one_to_one')
 assert len(sub)==3000 and sub.id.is_unique and sub.label.notna().all()
 assert list(sub.columns)==['id','label'] and sub.id.tolist()==sample.id.tolist()
 sub.to_csv(DEST/name,index=False)
 reloaded=pd.read_csv(DEST/name);assert reloaded.shape==(3000,2) and reloaded.label.between(0,1).all()
 return hashlib.sha256((DEST/name).read_bytes()).hexdigest()
h=save('submission_improved.csv',tc[chosen]);save('submission_neural_mlp.csv',b.mlp)
results.to_csv(DEST/'validation_comparison.csv',index=False)
for name in ['lexical_tuning.csv','style_results.csv','mlp_results.csv','mlp_summary.json']:
 shutil.copyfile(OUT/name,DEST/name)
summary={'selected':chosen,'selected_validation_auroc':float(results.iloc[0].validation_auc),'neural_mlp_validation_auroc':float(roc_auc_score(v.label,v.mlp)),'prior_character_svm_validation_auroc':0.969563,'prior_public_auroc_user_reported':0.85,'new_kaggle_score':'not yet measured','validation_rows':2000,'split_seed':42,'train_rows_available_for_refit':10000,'style_features':int(x.shape[1]),'neural_architecture':best,'mlp_ensemble_seeds':[42,43],'submission_rows':3000,'submission_sha256':h,'selection_caveat':'Same fixed validation set used for comparing hyperparameters and ensemble weights; not an independent test estimate. Hidden test distribution may differ.','transformer_pilot':'DistilBERT loaded and top two layers began fine-tuning; stopped after pilot due to about 2 seconds per batch. No completed transformer validation score or transformer predictions are claimed.'}
(DEST/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
