"""Frozen local language-model features; no labels are used by this script."""
import sys, os, argparse, time, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='openai-community/gpt2')
    p.add_argument('--download',action='store_true')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path)
    p.add_argument('--model-cache',type=Path,default=ROOT/'model_cache')
    args=p.parse_args()
    cache=args.model_cache;cache.mkdir(parents=True,exist_ok=True)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    dtype=torch.float16 if device=='cuda' else torch.float32
    print('Loading',args.model,'download allowed:',args.download,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model,cache_dir=str(cache),local_files_only=not args.download)
    model=AutoModelForCausalLM.from_pretrained(args.model,cache_dir=str(cache),local_files_only=not args.download,use_safetensors=True,torch_dtype=dtype)
    if args.download:
        print('Model/tokenizer cached successfully.',flush=True)
        return
    torch.set_num_threads(4)
    model=model.eval().to(device)
    tokenizer.pad_token=tokenizer.eos_token
    outdir=args.output or ROOT/'lm_features'/args.model.split('/')[-1];outdir.mkdir(parents=True,exist_ok=True)
    data=args.data
    for filename in ['train.csv','unlabeled.csv','test.csv']:
        frame=pd.read_csv(data/filename)
        out=outdir/filename
        if out.exists():
            existing=pd.read_csv(out)
            meta_path=outdir/(filename+'.json')
            cached_meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
            expected_hash=hashlib.sha256((data/filename).read_bytes()).hexdigest()
            if len(existing)==len(frame) and existing.id.tolist()==frame.id.tolist() and cached_meta.get('input_sha256')==expected_hash and cached_meta.get('model')==args.model:
                print('Already extracted',filename,flush=True);continue
        texts=frame.text.astype(str).tolist()
        tokens=tokenizer(texts,add_special_tokens=False,truncation=True,max_length=511)['input_ids']
        # Length buckets reduce padding while preserving original IDs in the saved table.
        order=sorted(range(len(tokens)),key=lambda i:len(tokens[i]))
        rows={};start=time.perf_counter()
        with torch.inference_mode():
            for offset in range(0,len(order),args.batch_size):
                ix=order[offset:offset+args.batch_size]
                ids=[[tokenizer.bos_token_id]+tokens[i] for i in ix]
                lengths=[len(a) for a in ids];width=max(lengths)
                input_ids=torch.full((len(ix),width),tokenizer.pad_token_id,dtype=torch.long,device=device)
                mask=torch.zeros_like(input_ids)
                for j,a in enumerate(ids):
                    input_ids[j,:len(a)]=torch.tensor(a,device=device);mask[j,:len(a)]=1
                logits=model(input_ids=input_ids,attention_mask=mask,use_cache=False).logits[:,:-1].float()
                labels=input_ids[:,1:]
                lp=torch.log_softmax(logits,dim=-1)
                target_lp=lp.gather(-1,labels.unsqueeze(-1)).squeeze(-1)
                entropy=-(lp.exp()*lp).sum(-1)
                target_logits=logits.gather(-1,labels.unsqueeze(-1))
                ranks=(logits>=target_logits).sum(-1).float()
                target_lp=target_lp.cpu().numpy();entropy=entropy.cpu().numpy();ranks=ranks.cpu().numpy()
                for j,i in enumerate(ix):
                    n=lengths[j]-1;a=target_lp[j,:n].astype(float);e=entropy[j,:n].astype(float);r=ranks[j,:n].astype(float)
                    f={'id':int(frame.id.iloc[i]),'lm_tokens':n,'lm_logprob_mean':a.mean(),'lm_logprob_std':a.std(),'lm_logprob_min':a.min(),'lm_entropy_mean':e.mean(),'lm_entropy_std':e.std(),'lm_surprisal_entropy':-a.mean()/max(e.mean(),1e-6),'lm_logrank_mean':np.log(r).mean(),'lm_rank_mean':r.mean()}
                    for q in [10,25,50,75,90]:f['lm_logprob_q'+str(q)]=np.percentile(a,q)
                    for k in [1,5,10,50]:f['lm_top'+str(k)+'_fraction']=np.mean(r<=k)
                    f['lm_surprisal_std']=(-a).std()
                    f['lm_lowprob_fraction']=np.mean(a<-8)
                    half=max(n//2,1);f['lm_half_drift']=a[half:].mean()-a[:half].mean() if n>half else 0.
                    f['lm_char_nll']=-a.sum()/max(len(texts[i]),1)
                    f['lm_adjacent_delta']=np.abs(np.diff(a)).mean() if n>1 else 0.
                    rows[i]=f
                del logits,lp,target_logits
                if offset//args.batch_size%25==0 or offset+args.batch_size>=len(order):
                    done=min(offset+args.batch_size,len(order));elapsed=time.perf_counter()-start
                    print(filename,done,'/',len(order),'elapsed',round(elapsed,1),'sec',flush=True)
                    pd.DataFrame([rows[i] for i in sorted(rows)]).to_csv(outdir/(filename+'.partial'),index=False)
        result=pd.DataFrame([rows[i] for i in range(len(frame))])
        assert result.id.tolist()==frame.id.tolist() and np.isfinite(result.drop(columns='id').to_numpy()).all()
        result.to_csv(out,index=False)
        metadata={'model':args.model,'max_tokens':512,'rows':len(frame),'input_sha256':hashlib.sha256((data/filename).read_bytes()).hexdigest(),'torch':torch.__version__,'seconds':time.perf_counter()-start,'features':[c for c in result if c!='id']}
        (outdir/(filename+'.json')).write_text(json.dumps(metadata,indent=2))
        print('Saved',out,flush=True)

if __name__=='__main__':main()
