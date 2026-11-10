"""Select among documented blends and verify the official Kaggle CSV format."""
from pathlib import Path
import json,zipfile,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from config import OUT, DATA_ZIP, PRIOR_RESULTS, DEST
v=pd.read_csv(PRIOR_RESULTS/'improved_validation.csv')[['id','label','selected_probability_ai']].rename(columns={'selected_probability_ai':'old'})
t=pd.read_csv(PRIOR_RESULTS/'deliverables/submission_improved.csv').rename(columns={'label':'old'})
protocol=pd.read_csv(OUT/'validation_protocol.csv');v=v.merge(protocol[['id','hard']],on='id',validate='one_to_one')
cv=pd.read_csv(OUT/'cnn/validation.csv');ct=pd.read_csv(OUT/'cnn/test.csv');v=v.merge(cv[['id','cnn']],on='id',validate='one_to_one');t=t.merge(ct,on='id',validate='one_to_one')
tcomp=pd.read_csv(OUT/'tree_comparison.csv');tcomp['robust_score']=(tcomp.validation_auc+tcomp.testlike_auc)/2
# Select up to three enriched feature tree variants, not old-feature control.
names=['xgb_rich3','xgb_rich5','cat_rich6']  # Frozen team-selected tree components
tv=pd.read_csv(OUT/'tree_validation.csv');tt=pd.read_csv(OUT/'tree_test.csv')
tv['tree']=tv[names].mean(axis=1);tt['tree']=tt[names].mean(axis=1);v=v.merge(tv[['id','tree']],on='id',validate='one_to_one');t=t.merge(tt[['id','tree']],on='id',validate='one_to_one')
pcomp=pd.read_csv(OUT/'pseudo_comparison.csv');bestp=pcomp.iloc[np.argmax((pcomp.validation_auc+pcomp.testlike_auc).to_numpy())];parent=tcomp[tcomp.model==bestp.parent].iloc[0]
pseudo_ok=bool(bestp.validation_auc>parent.validation_auc and bestp.testlike_auc>parent.testlike_auc)
pv=pd.read_csv(OUT/'pseudo_validation.csv');pt=pd.read_csv(OUT/'pseudo_test.csv')
v=v.merge(pv[['id',bestp.model]].rename(columns={bestp.model:'pseudo'}),on='id',validate='one_to_one');t=t.merge(pt[['id',bestp.model]].rename(columns={bestp.model:'pseudo'}),on='id',validate='one_to_one')
weights={'old':{'old':1.},'cnn_blend':{'old':.65,'cnn':.35},'tree_only':{'tree':1.},'old50_cnn30_tree20':{'old':.5,'cnn':.3,'tree':.2},'old35_cnn35_tree30':{'old':.35,'cnn':.35,'tree':.3}}
if pseudo_ok:
 weights['old50_cnn25_tree15_pseudo10']={'old':.5,'cnn':.25,'tree':.15,'pseudo':.1}
 weights['old35_cnn30_tree20_pseudo15']={'old':.35,'cnn':.3,'tree':.2,'pseudo':.15}
rows=[];pred={}
for name,w in weights.items():
 p=sum(v[k].to_numpy()*a for k,a in w.items());pred[name]=p;auc=roc_auc_score(v.label,p);hard=roc_auc_score(v.label[v.hard],p[v.hard]);rows.append({'model':name,'validation_auc':auc,'testlike_auc':hard,'selection_score':.5*(auc+hard)})
r=pd.DataFrame(rows).sort_values('selection_score',ascending=False);chosen=r.iloc[0].model
# Three predeclared small contributions from the independently fine-tuned compact BERT.
if (OUT/'minibert/neural_test.csv').exists():
 bv=pd.read_csv(OUT/'minibert/neural_validation.csv')[['id','probability_ai']].rename(columns={'probability_ai':'bert'})
 bt=pd.read_csv(OUT/'minibert/neural_test.csv')[['id','neural']].rename(columns={'neural':'bert'})
 v=v.merge(bv,on='id',validate='one_to_one');t=t.merge(bt,on='id',validate='one_to_one')
 base_name=chosen;base_weights=weights[base_name].copy();base_pred=pred[base_name].copy()
 for alpha in [.05,.1,.2]:
  name=base_name+'_bert'+str(alpha);w={k:a*(1-alpha) for k,a in base_weights.items()};w['bert']=alpha;weights[name]=w
  prob=(1-alpha)*base_pred+alpha*v.bert.to_numpy();pred[name]=prob
  auc=roc_auc_score(v.label,prob);ha=roc_auc_score(v.label[v.hard],prob[v.hard])
  rows.append({'model':name,'validation_auc':auc,'testlike_auc':ha,'selection_score':.5*(auc+ha)})
 r=pd.DataFrame(rows).sort_values('selection_score',ascending=False);chosen=r.iloc[0].model

# Preserve the exact selected model instead of reselecting weights on rerun.
chosen='old35_cnn35_tree30_bert0.2'
weights[chosen]={'old': 0.27999999999999997, 'cnn': 0.27999999999999997, 'tree': 0.24, 'bert': 0.2}
if 'bert' not in t.columns:raise RuntimeError('The selected final model requires BERT predictions.')
chosen_row=r.set_index('model').loc[chosen]

with zipfile.ZipFile(DATA_ZIP) as z:sample=pd.read_csv(z.open('sample_submission.csv'));test=pd.read_csv(z.open('test.csv'))
p=sum(t[k].to_numpy()*a for k,a in weights[chosen].items());data=pd.DataFrame({'id':t.id,'label':p});data=sample[['id']].merge(data,on='id',how='left',sort=False,validate='one_to_one')
assert data.shape==(3000,2) and data.id.is_unique and set(data.id)==set(test.id)
assert data.id.tolist()==sample.id.tolist() and np.isfinite(data.label).all() and data.label.between(0,1).all()
path=DEST/'submission_recommended.csv';data.to_csv(path,index=False);check=pd.read_csv(path);assert np.allclose(check.label,data.label,atol=1e-15,rtol=0)
r.to_csv(DEST/'validation_comparison.csv',index=False)
v['recommended_probability_ai']=pred[chosen];v.to_csv(OUT/'recommended_validation.csv',index=False)
summary={'chosen':chosen,'weights':weights[chosen],'tree_components':names,'validation_auc':float(chosen_row.validation_auc),'testlike_validation_auc':float(chosen_row.testlike_auc),'selection_rule':'Frozen team-selected final components and weights, chosen November 9','pseudo_included':bool('pseudo' in weights[chosen]),'pseudo_improved_both_checks':pseudo_ok,'new_kaggle_score':None,'prior_public_score_user_reported_rounded':.96,'heldout_authorship_labels_accessed':False,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':3000,'validation_seed':42,'external_pretrained_tool':'NLTK English POS tagger and base prajjwal1/bert-mini pretrained weights; only supplied competition labels used for authorship training','caveat':'Repeated model selection on fixed validation and test-like subset causes optimistic estimates. Test text is used without authorship labels only for auxiliary test-likeness grouping; no final feature extractor uses test labels.'}
(DEST/'results.json').write_text(json.dumps(summary,indent=2));print(r.to_string(index=False));print(json.dumps(summary,indent=2))
