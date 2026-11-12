"""Select format-robust blends with native AUC >= .994.

Selection objective fixed before executing this script: maximize the mean of
native AUC and the worst AUC across three synthetic whitespace transformations,
subject to native AUC >= .994. Not an estimate of hidden-test AUC.
"""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from advanced_features import expanded_features,canonical,reflow,load_lm_matrix
from predict_optimized import component_predict
from search_optimization import model_for,get_features
from train import ROOT,SEED,classifier


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--search',default='search');ap.add_argument('--name',default='robust_v1');ap.add_argument('--final',action='store_true');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization'
    df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy();cache=joblib.load(art/(args.search+'.joblib'))
    lexical=joblib.load(art/'lexical.joblib');tr=cache['tr'];va=cache['va'];variants=['raw','flat','single','double']
    baseline_components=joblib.load(art/'baseline_public_0.96962.joblib')
    original_cache=joblib.load(ROOT/'artifacts/validation.joblib')
    baseline_cache=art/'baseline_validation_variants.joblib'
    if baseline_cache.exists():baseline=joblib.load(baseline_cache)
    else:
        basemodels=[]
        for component in baseline_components:
            c=component.copy();kind=c['kind'];name={'word':'word_svm_1','char':'char_lr_16','style':'style_31_700'}[kind]
            c['model']=original_cache['models'][name]
            if kind!='style':c['vectorizer']=original_cache['matrices'][kind][0]
            basemodels.append(c)
        baseline={}
        transformations={'raw':lambda t:t,'flat':canonical,'single':lambda t:reflow(t,'\n'),'double':reflow}
        for name,f in transformations.items():baseline[name]=component_predict({'kind':'baseline','components':basemodels},df.text.iloc[va].map(f).tolist())
        joblib.dump(baseline,baseline_cache)
    lex_best=max(lexical['records'],key=lambda r:r['val_auc']);lex_pred=lexical['predictions'][lex_best['name']]
    # At most the strongest canonical and augmented configuration from each family.
    selected=[]
    for view in ['canonical','augmented','native']:
        for lm in [False,True]:
            pool=[r for r in cache['records'] if r['config']['view']==view and bool(r['config'].get('lm'))==lm]
            if pool:selected.append(max(pool,key=lambda r:r['robust_mean']))
    candidates=[]
    for record in selected:
        for baseline_weight in [0,.25,.5,.75]:
            for style_fraction in [.5,.75,1.]:
                weights=[baseline_weight,(1-baseline_weight)*style_fraction,(1-baseline_weight)*(1-style_fraction)]
                preds={v:weights[0]*baseline[v]+weights[1]*cache['predictions'][record['name']][v]+weights[2]*lex_pred[v] for v in variants}
                aucs={v:roc_auc_score(y[va],p) for v,p in preds.items()}
                objective=(aucs['raw']+min(aucs[v] for v in variants[1:]))/2
                candidates.append({'rich_name':record['name'],'lexical_name':lex_best['name'],'weights':weights,'aucs':aucs,
                    'objective':objective,'eligible':bool(aucs['raw']>=.994)})
    eligible=[r for r in candidates if r['eligible']]
    if not eligible:raise RuntimeError('No candidate meets the native-AUC floor.')
    chosen=max(eligible,key=lambda r:r['objective'])
    result={'name':args.name,'selection_rule':'Native AUC >= .994; maximize average(native AUC, worst of flat/single/double AUC).',
        'selected':chosen,'all_candidates':candidates,'baseline_aucs':{v:roc_auc_score(y[va],p) for v,p in baseline.items()},
        'limitation':'Repeated holdout selection and synthetic transforms; no guarantee of hidden-domain robustness.'}
    (out/(args.name+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(chosen,indent=2),flush=True)
    if not args.final:return
    selected_config=next(r['config'] for r in cache['records'] if r['name']==chosen['rich_name'])
    features=get_features(df,art/'rich_train.joblib');view=selected_config['view'];views=variants if view=='augmented' else ['flat' if view=='canonical' else 'raw']
    matrices=features
    if selected_config.get('lm'):
        lm=load_lm_matrix(art,'train',selected_config.get('lm_dirs',['lm']));matrices={k:np.column_stack([x,lm]) for k,x in features.items()}
    xt=np.concatenate([matrices[v] for v in views]);yt=np.tile(y,len(views))
    rich=model_for(selected_config);rich.fit(xt,yt)
    vectorizer=lexical['vectorizers'][lex_best['config']['analyzer']]
    flat=df.text.map(canonical);xx=vectorizer.fit_transform(flat)
    lexmodel=classifier(lex_best['config']['kind'],lex_best['config']['C']);lexmodel.fit(xx,y)
    b,s,l=chosen['weights']
    components=[{'kind':'baseline','weight':b,'components':baseline_components},
                {'kind':'rich','weight':s,'config':selected_config,'model':rich},
                {'kind':'lexical','weight':l,'model':lexmodel,'vectorizer':vectorizer}]
    bundle={'name':args.name,'components':[c for c in components if c['weight']>0],'selection':chosen,'seed':SEED}
    joblib.dump(bundle,art/(args.name+'.joblib'),compress=3)
    print('Saved model',art/(args.name+'.joblib'),flush=True)


if __name__=='__main__':main()
