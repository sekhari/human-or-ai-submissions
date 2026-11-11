"""Create paired bootstrap comparisons, error tables and a static experiment plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    parser.add_argument('--out',type=Path,default=root/'results/yuchen/catboost_v1')
    parser.add_argument('--data',type=Path,default=root/'data')
    args=parser.parse_args()
    records=json.loads((args.out/'metrics.json').read_text())
    best=json.loads((args.out/'best_style.json').read_text())
    blend=json.loads((args.out/'blend_selection.json').read_text())
    # Validate the exact three-seed blend recipe used for the final CSV.
    left=pd.read_csv(args.out/f'val_{blend["left"]}_3seed.csv')
    right=pd.read_csv(args.out/f'val_{blend["right"]}_3seed.csv')
    np.testing.assert_array_equal(left.id,right.id)
    w=blend['stack_weight']
    final_blend=left.copy()
    final_blend['pred']=(1-w)*left.pred+w*right.pred
    final_blend.to_csv(args.out/'val_cat_blend_3seed.csv',index=False)
    final_record={'model':'cat_blend_3seed','auc':float(roc_auc_score(final_blend.label,final_blend.pred)),
        'logloss':float(log_loss(final_blend.label,final_blend.pred)),
        'accuracy':float(accuracy_score(final_blend.label,final_blend.pred>=0.5)),
        'inner_auc':None,'seconds':None,'config':{'weight_selected_on_single_seed_inner_predictions':w}}
    records=[r for r in records if r['model']!='cat_blend_3seed']+[final_record]
    (args.out/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    names=['cat_style','cat_extended',best['model']+'_lr','cat_blend',
           'cat_pseudo_t0.98','cat_pseudo_t0.95',best['model']+'_3seed',
           best['model']+'_lr_3seed','cat_blend_3seed','reference_lgbm_lr']
    names=list(dict.fromkeys(n for n in names if (args.out/f'val_{n}.csv').exists()))
    predictions={n:pd.read_csv(args.out/f'val_{n}.csv') for n in names}
    ref=predictions['reference_lgbm_lr']
    for d in predictions.values():
        np.testing.assert_array_equal(d.id,ref.id)
        np.testing.assert_array_equal(d.label,ref.label)
    y=ref.label.to_numpy()
    rng=np.random.default_rng(20261110)
    positive=np.flatnonzero(y==1); negative=np.flatnonzero(y==0)
    boot={n:[] for n in names if n!='reference_lgbm_lr'}
    feature_deltas=[]
    for _ in range(1000):
        idx=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
        base=roc_auc_score(y[idx],ref.pred.to_numpy()[idx])
        for n in boot:
            boot[n].append(roc_auc_score(y[idx],predictions[n].pred.to_numpy()[idx])-base)
        feature_deltas.append(roc_auc_score(y[idx],predictions['cat_extended'].pred.to_numpy()[idx])-
                              roc_auc_score(y[idx],predictions['cat_style'].pred.to_numpy()[idx]))
    intervals=[]
    base_auc=roc_auc_score(y,ref.pred)
    for n,values in boot.items():
        low,high=np.quantile(values,[0.025,0.975])
        intervals.append({'model':n,'reference':'reference_lgbm_lr',
            'delta_auc':float(roc_auc_score(y,predictions[n].pred)-base_auc),
            'ci_95_low':float(low),'ci_95_high':float(high),'bootstrap_repeats':1000})
    low,high=np.quantile(feature_deltas,[0.025,0.975])
    intervals.append({'model':'cat_extended','reference':'cat_style',
        'delta_auc':float(roc_auc_score(y,predictions['cat_extended'].pred)-roc_auc_score(y,predictions['cat_style'].pred)),
        'ci_95_low':float(low),'ci_95_high':float(high),'bootstrap_repeats':1000})
    (args.out/'paired_bootstrap.json').write_text(json.dumps(intervals,indent=2)+'\n')
    train=pd.read_csv(args.data/'train.csv').set_index('id')
    rows=[]
    for name in [best['model'],best['model']+'_lr']:
        if name not in predictions: continue
        d=predictions[name].copy()
        d['error_type']=np.where((d.label==0)&(d.pred>=0.5),'FP',
                                np.where((d.label==1)&(d.pred<0.5),'FN','correct'))
        for error,ascending in [('FP',False),('FN',True)]:
            selected=d[d.error_type==error].sort_values('pred',ascending=ascending).head(5)
            for _,r in selected.iterrows():
                text=train.loc[r.id,'text']
                rows.append({'model':name,'id':int(r.id),'error_type':error,'probability':float(r.pred),
                             'word_count':len(text.split()),'has_blank_line':'\n\n' in text,
                             'excerpt':text[:250],'interpretation':'Pending human review; no inferred cause asserted.'})
    (args.out/'errors_for_review.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')

    plot_names=[n for n in names if n not in ['cat_blend',best['model']+'_3seed']]
    labels={
        'cat_style':'CatBoost: 53 style features',
        'cat_extended':'CatBoost: 90 extended features',
        best['model']+'_lr':'CatBoost + cross-fitted LR',
        'cat_pseudo_t0.98':'CatBoost + pseudo-labels (0.98)',
        'cat_pseudo_t0.95':'CatBoost + pseudo-labels (0.95)',
        best['model']+'_lr_3seed':'CatBoost + LR, 3 seeds',
        'cat_blend_3seed':'CatBoost blend, 3 seeds',
        'reference_lgbm_lr':'Team LightGBM comparator',
    }
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    values=[roc_auc_score(y,predictions[n].pred) for n in plot_names]
    colors=['#457b9d' if n!='reference_lgbm_lr' else '#8a8f98' for n in plot_names]
    bars=ax.barh([labels.get(n,n) for n in plot_names],values,color=colors)
    ax.invert_yaxis()
    ax.set_xlim(max(0,min(values)-0.002),1.0005)
    for bar,value in zip(bars,values):
        ax.text(value+0.00003,bar.get_y()+bar.get_height()/2,f'{value:.6f}',va='center',fontsize=9)
    ax.set_xlabel('AUROC (zoomed axis; fixed 2,000-row validation)')
    ax.set_title('Group 19 — CPU CatBoost experiments\nLocal validation only; no Kaggle score claim')
    ax.grid(axis='x',alpha=0.2); ax.set_axisbelow(True)
    fig.savefig(args.out/'validation_comparison.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'comparisons':len(intervals),'error_examples':len(rows),
                      'figure':str(args.out/'validation_comparison.png')},indent=2))


if __name__=='__main__': main()
