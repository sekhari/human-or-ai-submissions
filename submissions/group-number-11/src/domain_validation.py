"""Additional grouped validation using unlabeled lexical topic/style clusters.

Clusters are only observable proxies, not known source/domain/generator IDs.
No hidden test text or labels participate in this protocol.
"""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from advanced_features import canonical,load_lm_matrix
from search_optimization import get_features,model_for
from train import ROOT,SEED


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--lm-dirs',nargs='*');ap.add_argument('--name',default='domain');args=ap.parse_args()
    art=ROOT/'artifacts/optimization';out=ROOT/'results/optimization';df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy()
    cluster_path=art/'topic_clusters.npz'
    if cluster_path.exists():groups=np.load(cluster_path)['groups']
    else:
        v=TfidfVectorizer(max_features=20000,ngram_range=(1,1),min_df=5,stop_words='english',sublinear_tf=True)
        x=v.fit_transform(df.text.map(canonical));x=TruncatedSVD(n_components=48,n_iter=3,random_state=SEED).fit_transform(x)
        groups=MiniBatchKMeans(n_clusters=12,random_state=SEED,n_init=5,batch_size=512).fit_predict(x)
        np.savez_compressed(cluster_path,groups=groups,ids=df.id.to_numpy())
    features=get_features(df,art/'rich_train.joblib');lm=load_lm_matrix(art,'train',args.lm_dirs) if args.lm_dirs else None
    records=[]
    configs=[{'view':'native','kind':'lgb','leaves':31},{'view':'augmented','kind':'hist'},
             {'view':'canonical','kind':'lgb','leaves':15}]
    if lm is not None:configs.extend([{'view':'native','kind':'lgb','leaves':15,'lm':True},
                                      {'view':'canonical','kind':'lgb','leaves':15,'lm':True}])
    for fold,(tr,va) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(df,y,groups)):
        for config in configs:
            views=['raw','flat','single','double'] if config['view']=='augmented' else ['flat' if config['view']=='canonical' else 'raw']
            xx={k:np.column_stack([x,lm]) if config.get('lm') else x for k,x in features.items()}
            xt=np.concatenate([xx[k][tr] for k in views]);yt=np.tile(y[tr],len(views));start=time.perf_counter()
            m=model_for(config);m.fit(xt,yt)
            view='flat' if config['view']=='canonical' else 'raw';p=m.predict_proba(xx[view][va])[:,1]
            row={'fold':fold,'config':config,'train_rows':len(tr),'validation_rows':len(va),
                'heldout_clusters':np.unique(groups[va]).tolist(),'auc':roc_auc_score(y[va],p),'seconds':time.perf_counter()-start}
            records.append(row);print(json.dumps(row),flush=True)
            (out/(args.name+'.json')).write_text(json.dumps({'records':records,
                'limitation':'Topic/style clusters are an unsupervised proxy; not external source validation.'},indent=2))


if __name__=='__main__':main()
