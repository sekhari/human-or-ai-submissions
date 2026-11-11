"""CPU CatBoost experiments; preserves teammates' training code.

Fixed outer validation = fold 0 of the team's stratified 5-fold split.
All model/iteration/ensemble-weight selection uses an INTERNAL holdout in
the remaining 8,000 rows. Outer labels are used only for reporting.
Commands: --stage style|stack|pseudo|final|all. See YUCHEN_EXPERIMENTS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from baseline_tfidf import build_pipeline
from folds import SEED, make_folds
from yuchen_features import features

ROOT = Path(__file__).resolve().parents[2]
GRID = [(4,3), (4,10), (6,3), (6,10), (8,3), (8,10)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def metrics(y, p):
    p = np.asarray(p)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities')
    return dict(auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                logloss=float(log_loss(y,p,labels=[0,1])),
                accuracy=float(accuracy_score(y,p>=0.5)))


def signature(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()


def normal_text(t):
    return re.sub(r'\s+',' ',t).strip().lower()


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out/'cache'
        self.cache.mkdir(exist_ok=True)
        self.data = {name:pd.read_csv(args.data/f'{name}.csv') for name in ['train','unlabeled','test']}
        fingerprints = {}
        audit = {}
        for name,d in self.data.items():
            need = ['id','text']+(['label'] if name=='train' else [])
            if not set(need).issubset(d.columns) or d[need].isna().any().any() or d.id.duplicated().any():
                raise ValueError(f'Invalid data: {name}')
            if not d.text.map(lambda t:isinstance(t,str) and bool(t.strip())).all():
                raise ValueError(f'Empty/non-string texts: {name}')
            fingerprints[name] = signature(d[need])
            audit[name] = {'rows':len(d),'duplicate_texts':int(d.text.duplicated().sum())}
        self.train = self.data['train']
        self.y = self.train.label.to_numpy(dtype=int)
        if set(self.y)!={0,1}: raise ValueError('Expected binary labels')
        meta = {'fingerprints':fingerprints, 'seed':SEED, 'outer_fold':0, 'inner_lr_folds':3,
                'iterations_cap':args.iterations, 'grid':GRID, 'threads':args.threads,
                'python':platform.python_version(), 'versions':{p:importlib.metadata.version(p) for p in
                ['numpy','pandas','scikit-learn','catboost','lightgbm']},
                'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ['yuchen_experiments.py','yuchen_features.py','baseline_tfidf.py','style_features.py','folds.py']}}
        meta_path = self.out/'manifest.json'
        if meta_path.exists() and json.loads(meta_path.read_text())!=meta:
            raise ValueError('Data/config/code/environment changed. Use a new --out directory.')
        write_json(meta_path,meta)
        self.fold = make_folds(self.train,path=self.out/'folds.csv')
        self.tr = np.flatnonzero(self.fold!=0)
        self.va = np.flatnonzero(self.fold==0)
        self.fit,self.stop = train_test_split(self.tr,test_size=0.2,stratify=self.y[self.tr],random_state=SEED)
        roles = np.full(len(self.train),'outer_validation',dtype=object)
        roles[self.fit]='inner_fit'; roles[self.stop]='inner_tune'
        pd.DataFrame({'id':self.train.id,'role':roles,'fold':self.fold}).to_csv(self.out/'split.csv',index=False)
        canon = {name:set(d.text.map(normal_text)) for name,d in self.data.items()}
        audit['cross_split_text_overlap']={f'{a}_{b}':len(canon[a]&canon[b]) for a,b in
                                         [('train','unlabeled'),('train','test'),('unlabeled','test')]}
        audit['outer_train_validation_overlap']=len(set(self.train.iloc[self.tr].text.map(normal_text)) &
                                                  set(self.train.iloc[self.va].text.map(normal_text)))
        write_json(self.out/'data_audit.json',audit)
        # The prescribed stratified split is retained. Report duplicates explicitly.
        print('DATA AUDIT',json.dumps(audit),flush=True)
        fpath = self.cache/'features.joblib'
        if fpath.exists(): self.F=joblib.load(fpath)
        else:
            self.F={}
            for name,d in self.data.items():
                b,e=features(d.text)
                self.F[name]={'style':b,'extended':e}
            joblib.dump(self.F,fpath)
        print('FEATURES', {k:v.shape[1] for k,v in self.F['train'].items()},flush=True)

    def model(self, config, seed=SEED, iterations=None):
        return CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',
            iterations=iterations or self.args.iterations,learning_rate=0.03,
            depth=config['depth'],l2_leaf_reg=config['l2_leaf_reg'],
            random_seed=seed,thread_count=self.args.threads,allow_writing_files=False,
            verbose=False)

    def lr(self, indices, targets, tag):
        """Strict fold-local vectorizers; target rows never contribute to fitting."""
        path=self.cache/f'lr_{tag}.joblib'
        if path.exists(): return joblib.load(path)
        text=self.train.iloc[indices].text.reset_index(drop=True)
        y=self.y[indices]
        oof=np.empty(len(indices))
        cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
        for k,(tr,va) in enumerate(cv.split(text,y)):
            print(f'LR {tag} crossfit {k+1}/3',flush=True)
            pipe=build_pipeline(0,C=30)
            pipe.fit(text.iloc[tr],y[tr])
            oof[va]=pipe.predict_proba(text.iloc[va])[:,1]
        print(f'LR {tag} refit',flush=True)
        pipe=build_pipeline(0,C=30).fit(text,y)
        result={'oof':oof,'targets':{name:pipe.predict_proba(texts)[:,1] for name,texts in targets.items()}}
        joblib.dump(result,path)
        return result

    def frames(self, representation, stacked=False):
        f=self.F['train'][representation]
        a,b,c,d=[f.iloc[i].reset_index(drop=True).copy() for i in [self.fit,self.stop,self.tr,self.va]]
        if stacked:
            inner=self.lr(self.fit,{'stop':self.train.iloc[self.stop].text},'inner')
            outer=self.lr(self.tr,{'val':self.train.iloc[self.va].text,
                           'unlabeled':self.data['unlabeled'].text},'outer')
            a['lr_score']=inner['oof']; b['lr_score']=inner['targets']['stop']
            c['lr_score']=outer['oof']; d['lr_score']=outer['targets']['val']
        return a,b,c,d

    def evaluate(self,name,p,train_p=None,inner_auc=None,seconds=None,config=None):
        record={'model':name,**metrics(self.y[self.va],p),'inner_auc':inner_auc,
                'seconds':seconds,'config':config}
        if train_p is not None: record['train_auc']=float(roc_auc_score(self.y[self.tr],train_p))
        path=self.out/'metrics.json'
        rows=json.loads(path.read_text()) if path.exists() else []
        rows=[r for r in rows if r['model']!=name]+[record]
        write_json(path,rows)
        pd.DataFrame({'id':self.train.iloc[self.va].id.to_numpy(),'label':self.y[self.va],
                      'pred':p}).to_csv(self.out/f'val_{name}.csv',index=False)
        print('RESULT',json.dumps(record),flush=True)
        return record

    def search(self,representation,stacked=False):
        name='cat_'+representation+('_lr' if stacked else '')
        summary_path=self.out/f'{name}_selection.json'
        if summary_path.exists(): return json.loads(summary_path.read_text())
        a,b,c,d=self.frames(representation,stacked)
        records=[]
        best=None
        for depth,l2 in GRID:
            config={'depth':depth,'l2_leaf_reg':l2}
            t=time.perf_counter()
            model=self.model(config)
            model.fit(a,self.y[self.fit],eval_set=(b,self.y[self.stop]),early_stopping_rounds=150)
            p=model.predict_proba(b)[:,1]
            row={**config,'iterations':int(model.tree_count_),
                 'inner_auc':float(roc_auc_score(self.y[self.stop],p)),
                 'seconds':time.perf_counter()-t}
            records.append(row)
            print('TUNE',name,json.dumps(row),flush=True)
            if best is None or row['inner_auc']>best['inner_auc']:
                best=row
                pd.DataFrame({'id':self.train.iloc[self.stop].id.to_numpy(),
                              'label':self.y[self.stop],'pred':p}).to_csv(self.out/f'inner_{name}.csv',index=False)
        t=time.perf_counter()
        model=self.model(best,iterations=best['iterations'])
        model.fit(c,self.y[self.tr])
        p=model.predict_proba(d)[:,1]
        result=self.evaluate(name,p,model.predict_proba(c)[:,1],best['inner_auc'],
                             sum(r['seconds'] for r in records)+time.perf_counter()-t,best)
        model.save_model(str(self.out/f'{name}.cbm'))
        pd.DataFrame({'feature':c.columns,'importance':model.feature_importances_}).sort_values(
            'importance',ascending=False).to_csv(self.out/f'importance_{name}.csv',index=False)
        result.update(representation=representation,stacked=stacked,grid=records)
        write_json(summary_path,result)
        return result

    def style(self):
        choices=[self.search('style'),self.search('extended')]
        best=max(choices,key=lambda x:x['inner_auc'])
        write_json(self.out/'best_style.json',best)

    def stack(self):
        self.style()
        best=json.loads((self.out/'best_style.json').read_text())
        self.search(best['representation'],True)
        outer=self.lr(self.tr,{'val':self.train.iloc[self.va].text,
                      'unlabeled':self.data['unlabeled'].text},'outer')
        self.evaluate('tfidf_lr',outer['targets']['val'])
        # Original LightGBM function/config are used only as a comparator.
        from lgbm_stack import fit_lgbm
        for stack in [False,True]:
            name='reference_lgbm'+('_lr' if stack else '')
            if (self.out/f'val_{name}.csv').exists(): continue
            a=self.F['train']['style'].iloc[self.tr].reset_index(drop=True).copy()
            b=self.F['train']['style'].iloc[self.va].reset_index(drop=True).copy()
            if stack:
                a['lr_score']=outer['oof']; b['lr_score']=outer['targets']['val']
            t=time.perf_counter()
            p,model=fit_lgbm(a,self.y[self.tr],b)
            self.evaluate(name,p,seconds=time.perf_counter()-t,
                          config={'reference':'unchanged lgbm_stack.fit_lgbm','strict_lr_crossfit':True})
        self.blend()

    def blend(self):
        best=json.loads((self.out/'best_style.json').read_text())
        left=best['model']; right=left+'_lr'
        a=pd.read_csv(self.out/f'inner_{left}.csv'); b=pd.read_csv(self.out/f'inner_{right}.csv')
        if not a.id.equals(b.id): raise ValueError('Misaligned inner predictions')
        rows=[]
        for w in [0,0.25,0.5,0.75,1]:
            p=(1-w)*a.pred.to_numpy()+w*b.pred.to_numpy()
            rows.append({'stack_weight':w,'inner_auc':float(roc_auc_score(a.label,p))})
        chosen=max(rows,key=lambda x:x['inner_auc'])
        av=pd.read_csv(self.out/f'val_{left}.csv'); bv=pd.read_csv(self.out/f'val_{right}.csv')
        if not av.id.equals(bv.id): raise ValueError('Misaligned predictions')
        w=chosen['stack_weight']
        self.evaluate('cat_blend',(1-w)*av.pred.to_numpy()+w*bv.pred.to_numpy(),
                      inner_auc=chosen['inner_auc'],config={'left':left,'right':right,**chosen})
        write_json(self.out/'blend_selection.json',{'left':left,'right':right,**chosen,'grid':rows})
        # LightGBM blends are exploratory fixed-validation comparisons, not selected submissions.
        lgb=pd.read_csv(self.out/'val_reference_lgbm_lr.csv')
        if not av.id.equals(lgb.id): raise ValueError('Misaligned reference predictions')
        for w in [0.1,0.2,0.3,0.5]:
            self.evaluate(f'exploratory_lgbm_cat_w{w}',(1-w)*lgb.pred.to_numpy()+w*bv.pred.to_numpy(),
                          config={'cat_weight':w,'selection_warning':'exploratory; fixed validation used for comparison'})

    def pseudo(self):
        self.style()
        best=json.loads((self.out/'best_style.json').read_text())
        rep=best['representation']; config=best['config']; name=best['model']
        model=CatBoostClassifier(); model.load_model(str(self.out/f'{name}.cbm'))
        u=self.data['unlabeled']; Fu=self.F['unlabeled'][rep]
        pu=model.predict_proba(Fu)[:,1]
        # Exclude repeated labeled text without using validation labels or fitting on validation text.
        seen=set(self.train.text.map(normal_text))
        eligible=(~u.text.map(normal_text).isin(seen) & ~u.text.map(normal_text).duplicated()).to_numpy()
        for threshold in [0.98,0.95]:
            selected=eligible&((pu>=threshold)|(pu<=1-threshold))
            tag=f'cat_pseudo_t{threshold}'
            if (self.out/f'val_{tag}.csv').exists(): continue
            xp=pd.concat([self.F['train'][rep].iloc[self.tr],Fu.loc[selected]],ignore_index=True)
            yp=np.r_[self.y[self.tr],(pu[selected]>=0.5).astype(int)]
            weights=np.r_[np.ones(len(self.tr)),np.full(selected.sum(),0.25)]
            t=time.perf_counter()
            student=self.model(config,iterations=config['iterations'])
            student.fit(xp,yp,sample_weight=weights)
            p=student.predict_proba(self.F['train'][rep].iloc[self.va])[:,1]
            self.evaluate(tag,p,seconds=time.perf_counter()-t,config={
                'teacher':name,'threshold':threshold,'pseudo_weight':0.25,
                'selected':int(selected.sum()),'pseudo_positive':int((pu[selected]>=0.5).sum()),
                'iterations':config['iterations'],'selection_warning':'exploratory; not auto-selected for final'})
            pd.DataFrame({'id':u.id,'teacher_prob':pu,'selected':selected}).to_csv(self.out/f'{tag}_selection.csv',index=False)

    def final(self):
        self.stack()
        best=json.loads((self.out/'best_style.json').read_text())
        stack=json.loads((self.out/f'{best["model"]}_lr_selection.json').read_text())
        blend=json.loads((self.out/'blend_selection.json').read_text())
        full_lr=self.lr(np.arange(len(self.train)),{'test':self.data['test'].text},'full')
        test=self.data['test']
        predictions={}
        for choice in [best,stack]:
            tag=choice['model']
            if (self.out/f'submission_{tag}.csv').exists():
                predictions[tag]=pd.read_csv(self.out/f'submission_{tag}.csv').label.to_numpy(); continue
            a=self.F['train'][choice['representation']].copy()
            b=self.F['test'][choice['representation']].copy()
            if choice['stacked']:
                a['lr_score']=full_lr['oof']; b['lr_score']=full_lr['targets']['test']
            seeds=[SEED,SEED+1,SEED+2]
            pp=[]
            for seed in seeds:
                print('FINAL FIT',tag,seed,flush=True)
                model=self.model(choice['config'],seed=seed,iterations=choice['config']['iterations'])
                model.fit(a,self.y)
                pp.append(model.predict_proba(b)[:,1])
                model.save_model(str(self.out/f'final_{tag}_{seed}.cbm'))
            predictions[tag]=np.mean(pp,axis=0)
            self.submission(tag,predictions[tag])
        w=blend['stack_weight']
        self.submission('cat_blend',(1-w)*predictions[blend['left']]+w*predictions[blend['right']])
        write_json(self.out/'final_recipe.json',{'seeds':[SEED,SEED+1,SEED+2],
            'style':best['config'],'stack':stack['config'],'blend':blend,
            'note':'All 10000 labels used. 3-seed averaging is separately validated in stability stage.'})

    def submission(self,tag,p):
        test=self.data['test']
        if len(p)!=len(test) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError('Invalid submission')
        pd.DataFrame({'id':test.id,'label':p}).to_csv(self.out/f'submission_{tag}.csv',index=False)

    def stability(self):
        self.stack()
        best=json.loads((self.out/'best_style.json').read_text())
        for tag in [best['model'],best['model']+'_lr']:
            choice=json.loads((self.out/f'{tag}_selection.json').read_text())
            _,_,a,b=self.frames(choice['representation'],choice['stacked'])
            pp=[pd.read_csv(self.out/f'val_{tag}.csv').pred.to_numpy()]
            for seed in [SEED+1,SEED+2]:
                m=self.model(choice['config'],seed=seed,iterations=choice['config']['iterations'])
                m.fit(a,self.y[self.tr]); p=m.predict_proba(b)[:,1]; pp.append(p)
                self.evaluate(f'{tag}_seed{seed}',p,config={'seed':seed})
            self.evaluate(f'{tag}_3seed',np.mean(pp,axis=0),config={'seeds':[SEED,SEED+1,SEED+2]})

    def summarize(self):
        path=self.out/'metrics.json'
        if not path.exists(): return
        rows=json.loads(path.read_text())
        lines=['# CatBoost experiments — measured results','',
               'Fixed outer validation: team fold 0; internal tuning: 20% of outer-training rows.',
               'Hyperparameters, representation and CatBoost blend weight selected by INTERNAL AUROC.',
               'Pseudo-label and LightGBM blend comparisons are exploratory. Public/private scores are unknown.','',
               '| Model | Internal AUROC | Fixed validation AUROC | Log loss | Seconds |',
               '|---|---:|---:|---:|---:|']
        for r in rows:
            internal='—' if r['inner_auc'] is None else f'{r["inner_auc"]:.6f}'
            sec='—' if r['seconds'] is None else f'{r["seconds"]:.1f}'
            lines.append(f'| {r["model"]} | {internal} | {r["auc"]:.6f} | {r["logloss"]:.6f} | {sec} |')
        (self.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        # Observable subgroups only; no inferred test labels.
        subgroup_rows=[]
        vt=self.train.iloc[self.va]
        lengths=vt.text.str.split().str.len().to_numpy()
        groups={'short_<150':lengths<150,'medium_150_249':(lengths>=150)&(lengths<250),
                'long_250plus':lengths>=250,'blank_line':vt.text.str.contains(r'\n\s*\n',regex=True).to_numpy()}
        for r in rows:
            pred=pd.read_csv(self.out/f'val_{r["model"]}.csv').pred.to_numpy()
            for group,mask in groups.items():
                if mask.sum(): subgroup_rows.append({'model':r['model'],'group':group,'n':int(mask.sum()),
                     'positives':int(self.y[self.va][mask].sum()),**metrics(self.y[self.va][mask],pred[mask])})
        write_json(self.out/'subgroups.json',subgroup_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--out',type=Path,default=ROOT/'results/yuchen/catboost_v1')
    p.add_argument('--stage',choices=['style','stack','pseudo','stability','final','all'],default='all')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--iterations',type=int,default=1600)
    args=p.parse_args()
    with threadpool_limits(limits=args.threads):
        e=Experiment(args)
        try:
            if args.stage=='all':
                e.style(); e.stack(); e.pseudo(); e.stability(); e.final()
            else: getattr(e,args.stage)()
        finally: e.summarize()


if __name__=='__main__': main()
