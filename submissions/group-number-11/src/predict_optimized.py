"""Predict with the experimental rich-feature/LM bundle, preserving sample IDs."""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from advanced_features import expanded_features,canonical,combine_lm_matrices
from features import style_features,normalize
from train import ROOT,probability


def conservative_blend(new, baseline, texts, weight, method):
    if method=='conditional_rank':
        from scipy.stats import rankdata
        groups=np.array(['\n\n' in normalize(t) for t in texts])
        calibrated=new.copy()
        for group in [False,True]:
            mask=groups==group;size=int(mask.sum())
            if size<2:continue
            quantiles=(rankdata(new[mask],method='average')-.5)/size
            calibrated[mask]=np.quantile(baseline[mask],quantiles)
        new=calibrated
    return weight*baseline+(1-weight)*new


def component_predict(component, texts, lm=None):
    if component['kind']=='baseline':
        p=np.zeros(len(texts))
        for c in component['components']:
            if c['kind']=='style':x=style_features(texts)
            else:
                x=c['vectorizer'].transform(texts)
                if c['ratio'] is not None:x=x.multiply(c['ratio'])
            p+=c['weight']*probability(c['model'],x)
        return p
    if component['kind']=='lexical':
        return probability(component['model'],component['vectorizer'].transform([canonical(t) for t in texts]))
    x=expanded_features(texts,canonicalize=component['config']['view']=='canonical')
    if component['config'].get('lm'):
        if lm is None:raise ValueError('This model requires the local LM feature cache.')
        x=np.column_stack([x,lm])
    return probability(component['model'],x)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--test',type=Path,required=True);ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--lm-cache',type=Path,action='append');args=ap.parse_args()
    frame=pd.read_csv(args.test);sample=pd.read_csv(args.sample);bundle=joblib.load(args.model)
    assert frame.id.is_unique and sample.id.is_unique and set(frame.id)==set(sample.id)
    lm=None
    lm_components=[c for c in bundle['components'] if c.get('config',{}).get('lm')]
    files=args.lm_cache
    if lm_components and not files:
        files=[ROOT/'artifacts/optimization'/d/'test_gpt2.npz' for d in lm_components[0]['config'].get('lm_dirs',['lm'])]
    if files:
        matrices=[]
        for file in files:
            cache=np.load(file);assert str(cache['data_sha256'])==hashlib.sha256(args.test.read_bytes()).hexdigest()
            lookup=pd.Series(np.arange(len(cache['ids'])),index=cache['ids']);matrices.append(cache['features'][lookup.loc[frame.id].to_numpy()])
        lm=combine_lm_matrices(matrices)
    p=np.zeros(len(frame))
    for c in bundle['components']:p+=c['weight']*component_predict(c,frame.text.tolist(),lm)
    assert abs(sum(c['weight'] for c in bundle['components'])-1)<1e-9
    if bundle.get('conservative_blend'):
        blend=bundle['conservative_blend'];base=component_predict({'kind':'baseline','components':blend['baseline_components']},frame.text.tolist())
        p=conservative_blend(p,base,frame.text.tolist(),blend['baseline_weight'],blend['method'])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    result=sample[['id']].merge(pd.DataFrame({'id':frame.id,'label':p}),on='id',validate='one_to_one',sort=False)
    args.out.parent.mkdir(exist_ok=True,parents=True);result.to_csv(args.out,index=False)
    print('Saved',args.out,len(result),'probability range',float(p.min()),float(p.max()))


if __name__=='__main__':main()
