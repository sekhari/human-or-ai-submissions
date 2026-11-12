"""Locally fine-tune generic RoBERTa using only supplied authorship labels.

Lower eight encoder blocks and embeddings stay frozen; the upper four and
classification head learn from random token crops of canonicalized text.
The original fixed 2,000-row validation split chooses the checkpoint.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,log_loss
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from advanced_features import canonical
from train import ROOT,SEED

MODEL_ID='FacebookAI/roberta-base'
REVISION='e2da8e2f811d1448a5b465c236feacd80ffbac7b'


def batch_tokens(encoded,indices,tokenizer,length,device,mode='first',rng=None):
    arr=np.full((len(indices),length),tokenizer.pad_token_id,dtype=np.int64)
    for row,index in enumerate(indices):
        ids=encoded[index];width=length-2
        if mode=='random' and len(ids)>width:start=int(rng.integers(0,len(ids)-width+1))
        elif mode=='last':start=max(len(ids)-width,0)
        else:start=0
        ids=[tokenizer.bos_token_id]+ids[start:start+width]+[tokenizer.eos_token_id]
        arr[row,:len(ids)]=ids
    tokens=torch.from_numpy(arr).to(device)
    return {'input_ids':tokens,'attention_mask':(tokens!=tokenizer.pad_token_id).long()}


def predict(model,encoded,indices,tokenizer,length,device,batch_size=16):
    model.eval();out=[]
    with torch.inference_mode():
        for step in range(0,len(indices),batch_size):
            part=list(indices[step:step+batch_size]);valid=len(part)
            while len(part)<batch_size:part.append(part[-1])
            views=[]
            for mode in ['first','last']:
                batch=batch_tokens(encoded,part,tokenizer,length,device,mode)
                views.append(torch.softmax(model(**batch).logits.float(),dim=-1)[:valid,1].cpu().numpy())
            out.extend((views[0]+views[1])/2)
    return np.asarray(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--max-length',type=int,default=192)
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--train-last',type=int,default=4)
    ap.add_argument('--name',default='roberta_v1');ap.add_argument('--max-seconds',type=int,default=1200);ap.add_argument('--max-steps',type=int);args=ap.parse_args()
    seed=20261111;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(3)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps':torch.mps.set_per_process_memory_fraction(.65)
    art=ROOT/'artifacts/optimization'/args.name;art.mkdir(exist_ok=True,parents=True);out=ROOT/'results/optimization'
    df=pd.read_csv(args.data/'train.csv');y=df.label.to_numpy()
    tr,va=train_test_split(np.arange(len(df)),test_size=2000,stratify=y,random_state=SEED)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION)
    encoded=tokenizer(df.text.map(canonical).tolist(),add_special_tokens=False,truncation=True,max_length=1022)['input_ids']
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,revision=REVISION,num_labels=2).to(device)
    for p in model.roberta.embeddings.parameters():p.requires_grad=False
    for layer in model.roberta.encoder.layer[:-args.train_last]:
        for p in layer.parameters():p.requires_grad=False
    head=list(model.classifier.parameters());head_ids={id(p) for p in head}
    body=[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer=torch.optim.AdamW([{'params':body,'lr':2e-5},{'params':head,'lr':1e-4}],weight_decay=.01)
    steps_per_epoch=int(np.ceil(len(tr)/args.batch_size));total_steps=steps_per_epoch*args.epochs;warmup=max(int(.06*total_steps),1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/warmup,max((total_steps-step)/(total_steps-warmup),0)))
    rng=np.random.default_rng(seed);best=-1.;history=[];start=time.perf_counter();global_step=0
    budget_hit=False
    for epoch in range(args.epochs):
        order=rng.permutation(tr);model.train();losses=[]
        for step in range(steps_per_epoch):
            indices=order[step*args.batch_size:(step+1)*args.batch_size]
            batch=batch_tokens(encoded,indices,tokenizer,args.max_length,device,'random',rng)
            labels=torch.tensor(y[indices],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True);result=model(**batch,labels=labels);loss=result.loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(body+head,1.);optimizer.step();scheduler.step()
            losses.append(float(loss.detach().cpu()));global_step+=1
            if global_step%25==0:print(f'epoch={epoch+1} step={global_step}/{total_steps} loss={np.mean(losses[-25:]):.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            budget_hit=time.perf_counter()-start>=args.max_seconds or (args.max_steps is not None and global_step>=args.max_steps)
            if global_step%(steps_per_epoch//2)==0 or global_step==total_steps or budget_hit:
                pv=predict(model,encoded,va,tokenizer,args.max_length,device,args.batch_size)
                record={'epoch':epoch+1,'step':global_step,'val_auc':roc_auc_score(y[va],pv),
                        'log_loss':log_loss(y[va],pv),'training_loss':float(np.mean(losses)),
                        'seconds':time.perf_counter()-start}
                history.append(record);print('VALIDATION',json.dumps(record),flush=True)
                if record['val_auc']>best:
                    best=record['val_auc'];model.save_pretrained(art/'best');tokenizer.save_pretrained(art/'best')
                    np.savez_compressed(art/'validation.npz',ids=df.id.iloc[va].to_numpy(),labels=y[va],probability=pv)
                (out/(args.name+'.json')).write_text(json.dumps({'pretrained_model':MODEL_ID,'revision':REVISION,
                    'seed':seed,'split_seed':SEED,'training_rows':len(tr),'validation_rows':len(va),
                    'max_length':args.max_length,'train_last_blocks':args.train_last,'history':history,
                    'normalization':'canonical whitespace','inference':'mean of first and last token-window probabilities',
                    'best_validation_auc':best},indent=2))
                model.train()
            if budget_hit:break
        if device=='mps':torch.mps.empty_cache()
        if budget_hit:break
    del optimizer,model
    if device=='mps':torch.mps.empty_cache()
    best_model=AutoModelForSequenceClassification.from_pretrained(art/'best').to(device)
    test=pd.read_csv(args.data/'test.csv');test_encoded=tokenizer(test.text.map(canonical).tolist(),add_special_tokens=False,truncation=True,max_length=1022)['input_ids']
    pt=predict(best_model,test_encoded,np.arange(len(test)),tokenizer,args.max_length,device,args.batch_size)
    np.savez_compressed(art/'test.npz',ids=test.id.to_numpy(),probability=pt)
    print('Saved transformer test predictions',best,flush=True)


if __name__=='__main__':main()
