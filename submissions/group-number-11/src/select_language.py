"""Conservative language-feature selection after robust_v1's negative result.

Keep at least 25% of the 0.96962 baseline. Require native validation AUC >=
0.9975 and maximize .75*blank-line-free subgroup AUC + .25*overall native AUC.
Compare plain blending and label-free within-format quantile alignment.
Synthetic style-only LM stress is not used as end-to-end validation evidence.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from advanced_features import load_lm_matrix
from features import normalize
from predict_optimized import conservative_blend
from search_optimization import model_for,get_features
from train import ROOT,SEED


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--search',default='language_small');ap.add_argument('--name',default='language_v2');ap.add_argument('--final',action='store_true');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization';df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy()
    cache=joblib.load(art/(args.search+'.joblib'));tr=cache['tr'];va=cache['va'];texts=df.text.iloc[va].tolist();yy=y[va]
    if (art/'baseline_validation_variants.joblib').exists():baseline=joblib.load(art/'baseline_validation_variants.joblib')['raw']
    else:
        original=joblib.load(ROOT/'artifacts/validation.joblib')
        baseline=.125*original['predictions']['word_svm_1']+.125*original['predictions']['char_lr_16']+.75*original['predictions']['style_31_700']
    hard=np.array(['\n\n' not in normalize(t) for t in texts])
    # Compare the best native configuration from each representation/training view.
    short=[]
    for view in ['native','raw_flat','canonical']:
        pool=[r for r in cache['records'] if r['config']['view']==view]
        if pool:short.append(max(pool,key=lambda r:r['aucs']['raw']))
    candidates=[];valid_predictions={}
    for row in short:
        p=cache['predictions'][row['name']]['raw']
        for method in ['plain','conditional_rank']:
            for weight in [.25,.5,.75]:
                pv=conservative_blend(p,baseline,texts,weight,method)
                native=roc_auc_score(yy,pv);sub=roc_auc_score(yy[hard],pv[hard]);objective=.75*sub+.25*native
                item={'source':row['name'],'method':method,'baseline_weight':weight,'val_auc':native,
                      'blank0_auc':sub,'objective':objective,'eligible':bool(native>=.9975)}
                candidates.append(item);valid_predictions[len(candidates)-1]=pv
    eligible=[(i,r) for i,r in enumerate(candidates) if r['eligible']]
    if not eligible:raise RuntimeError('No language candidate meets the native-AUC floor.')
    chosen_index,chosen=max(eligible,key=lambda pair:pair[1]['objective'])
    config=next(r['config'] for r in cache['records'] if r['name']==chosen['source'])
    result={'selected':chosen,'config':config,'all_candidates':candidates,
        'baseline_native_auc':roc_auc_score(yy,baseline),'baseline_blank0_auc':roc_auc_score(yy[hard],baseline[hard]),
        'selection_rule':'At least 25% baseline, native AUC >= .9975; maximize .75*blank0 AUC + .25*overall native AUC.',
        'calibration_note':'conditional_rank maps each new-score rank to the baseline score distribution within the same observed blank-line group; it uses no true labels at inference.'}
    (out/(args.name+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(chosen,indent=2),flush=True)
    pd.DataFrame({'id':df.id.iloc[va].to_numpy(),'label':yy,'probability':valid_predictions[chosen_index]}).to_csv(art/(args.name+'_validation.csv'),index=False)
    if not args.final:return
    features=get_features(df,art/'rich_train.joblib');lm=load_lm_matrix(art,'train',config.get('lm_dirs',['lm'])) if config.get('lm') else None
    view=config['view'];views=['raw','flat'] if view=='raw_flat' else ['flat' if view=='canonical' else 'raw']
    matrices={v:np.column_stack([x,lm]) if lm is not None else x for v,x in features.items()}
    xt=np.concatenate([matrices[v] for v in views]);yt=np.tile(y,len(views));m=model_for(config);m.fit(xt,yt)
    baseline_model=joblib.load(art/'baseline_public_0.96962.joblib')
    bundle={'name':args.name,'seed':SEED,'components':[{'kind':'rich','weight':1.,'config':config,'model':m}],
        'conservative_blend':{'baseline_components':baseline_model,'baseline_weight':chosen['baseline_weight'],'method':chosen['method']},
        'selection':chosen}
    joblib.dump(bundle,art/(args.name+'.joblib'),compress=3);print('Saved',art/(args.name+'.joblib'),flush=True)


if __name__=='__main__':main()
