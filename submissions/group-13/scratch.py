"""Standalone training/inference around the original, unchanged BiGRU class."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import expit
from common import ROOT,write_json
sys.path.insert(0,str(ROOT/'experiments'))
import neural_network_experiments as original
def encode(texts,vocab,config):
    return [torch.tensor([vocab.get(t,1) for t in original.tokenize(text)[:config['max_tokens']]] or [1],dtype=torch.long) for text in texts]
def fit(train,validation,out,device,seed=42,pooling='attention',epochs=20,strong=False):
    config=dict(original.SETTINGS,max_epochs=epochs)
    if strong:config.update(embedding_dropout=.4,classifier_dropout=.6,weight_decay=.01)
    counts=Counter(t for text in train.text for t in set(original.tokenize(text)))
    vocab={'<PAD>':0,'<UNK>':1};vocab.update({t:i+2 for i,t in enumerate(sorted(t for t,n in counts.items() if n>=2))})
    if validation is not None:
        sequences={'train':encode(train.text,vocab,config),'validation':encode(validation.text,vocab,config)}
        original.fit_one(pooling,seed,config,vocab,sequences,train.label.to_numpy(),validation.label.to_numpy(),torch.device(device),out,'portable-entrypoint',False)
        source=out/f'bigru_{pooling}_seed{seed}'/'best_model.pt'
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        state=checkpoint['state_dict']
    else:
        original.seed_everything(seed);m=original.BiGRUClassifier(len(vocab),pooling,config).to(device)
        original.seed_everything(seed+10000);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=config['weight_decay'],foreach=False)
        seq=encode(train.text,vocab,config);rng=np.random.default_rng(seed+20000);y=train.label.to_numpy()
        for epoch in range(epochs):
            m.train();order=rng.permutation(len(train))
            for start in range(0,len(order),32):
                tokens,lengths,labels=original.batch_at(seq,y,order[start:start+32],torch.device(device))
                opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(m(tokens,lengths),labels)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
            print({'epoch':epoch+1},flush=True)
        state=original.cpu_state(m)
    torch.save(dict(state_dict=state,vocabulary=vocab,config=config,pooling=pooling),out/'scratch.pt')
    write_json(out/'scratch_config.json',dict(config=config,pooling=pooling,seed=seed,full_refit=validation is None))
def load(out,device):
    c=torch.load(out/'scratch.pt',map_location='cpu',weights_only=True)
    m=original.BiGRUClassifier(len(c['vocabulary']),c['pooling'],c['config']).to(device);m.load_state_dict(c['state_dict'])
    return m,c
def predict(out,texts,device):
    m,c=load(out,device);seq=encode(texts,c['vocabulary'],c['config'])
    return expit(original.predict(m,seq,np.zeros(len(seq)),torch.device(device)))
