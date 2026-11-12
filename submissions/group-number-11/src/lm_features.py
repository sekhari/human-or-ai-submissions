"""Local frozen GPT-2 language statistics, never external classification calls."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
from features import normalize

MODEL_ID='openai-community/gpt2'
NAMES=['nll_mean','nll_std','nll_q10','nll_q25','nll_q50','nll_q75','nll_q90','nll_q95',
       'nll_min','nll_max','nll_first_half','nll_second_half','nll_half_difference',
       'nll_adjacent_difference','nll_first32','nll_last32','nll_top1_fraction',
       'nll_top10_fraction','nll_top100_fraction','nll_above5_fraction',
       'nll_above10_fraction','token_count','token_per_word','logit_margin_mean',
       'logit_margin_std','nll_every16_mean']


def extract(texts, model, tokenizer, device, batch_size=4, max_tokens=256, checkpoint=None, signature=None):
    encoded=tokenizer([normalize(t) for t in texts],truncation=True,max_length=max_tokens,
                      add_special_tokens=False)['input_ids']
    features=np.zeros((len(texts),len(NAMES)),dtype=np.float32);done=np.zeros(len(texts),dtype=bool)
    if checkpoint is not None and checkpoint.exists():
        cached=np.load(checkpoint)
        if str(cached['signature'])==str(signature):features=cached['features'];done=cached['done'];print('Resuming',int(done.sum()),flush=True)
    order=np.asarray([i for i in np.argsort([len(t) for t in encoded]) if not done[i]])
    def save_checkpoint():
        if checkpoint is not None:
            temporary=Path(str(checkpoint)+'.tmp.npz');np.savez_compressed(temporary,features=features,done=done,signature=signature);temporary.replace(checkpoint)
    start=time.perf_counter();model.eval()
    for step in range(0,len(order),batch_size):
        ids=order[step:step+batch_size];length=min(max_tokens,64*((max(len(encoded[i]) for i in ids)+63)//64))
        tokens=torch.full((len(ids),length),tokenizer.eos_token_id,dtype=torch.long,device=device)
        mask=torch.zeros_like(tokens)
        for j,i in enumerate(ids):tokens[j,:len(encoded[i])]=torch.tensor(encoded[i],device=device);mask[j,:len(encoded[i])]=1
        with torch.inference_mode():
            logits=model(input_ids=tokens,attention_mask=mask,use_cache=False).logits[:,:-1].float()
            targets=tokens[:,1:]
            observed=logits.gather(-1,targets.unsqueeze(-1)).squeeze(-1)
            nll=torch.logsumexp(logits,dim=-1)-observed
            rank=(logits>observed.unsqueeze(-1)).sum(dim=-1,dtype=torch.int32)
            ranks=torch.stack([(rank<k).float() for k in [1,10,100]],dim=-1)
            margins=logits.amax(dim=-1)-observed
            nll=nll.cpu().numpy();ranks=ranks.cpu().numpy();margins=margins.cpu().numpy()
            del logits,rank,observed
        for j,i in enumerate(ids):
            size=len(encoded[i])-1;values=nll[j,:size];half=max(size//2,1)
            if size<2:continue
            first=values[:half].mean();second=values[half:].mean()
            row=[values.mean(),values.std(),*np.quantile(values,[.1,.25,.5,.75,.9,.95]),
                 values.min(),values.max(),first,second,second-first,np.abs(np.diff(values)).mean(),
                 values[:32].mean(),values[-32:].mean(),*ranks[j,:size].mean(axis=0),
                 (values>5).mean(),(values>10).mean(),size,size/max(len(normalize(texts[i]).split()),1),
                 margins[j,:size].mean(),margins[j,:size].std(),values[::16].mean()]
            features[i]=row
            done[i]=True
        if step%(batch_size*20)==0:
            print(f'LM {step}/{len(order)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        if step%(batch_size*40)==0:save_checkpoint()
        if device=='mps' and step%(batch_size*40)==0:torch.mps.empty_cache()
    save_checkpoint()
    return features


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--max-tokens',type=int,default=256);ap.add_argument('--revision')
    ap.add_argument('--model-id',default=MODEL_ID)
    ap.add_argument('--canonicalize',action='store_true')
    ap.add_argument('--datasets',nargs='+',default=['train','test','unlabeled'])
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    revision=args.revision or HfApi().model_info(args.model_id).sha
    tokenizer=AutoTokenizer.from_pretrained(args.model_id,revision=revision)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps':torch.mps.set_per_process_memory_fraction(.65)
    model=AutoModelForCausalLM.from_pretrained(args.model_id,revision=revision,
        dtype=torch.float16 if device=='mps' else torch.float32).to(device)
    metadata={'model':args.model_id,'revision':revision,'device':device,'max_tokens':args.max_tokens,
              'features':NAMES,'labels_used':False,'normalization':'canonical whitespace' if args.canonicalize else 'decode escaped newlines/tabs; retain original formatting'}
    metadata_path=args.output/'lm_metadata.json'
    if metadata_path.exists():
        previous=json.loads(metadata_path.read_text())
        keys=['model','revision','max_tokens','normalization']
        if any(previous.get(k)!=metadata.get(k) for k in keys) and any(args.output.glob('*_gpt2.npz')):
            raise ValueError('Existing complete LM cache uses different settings; choose a fresh output directory.')
    (args.output/'lm_metadata.json').write_text(json.dumps(metadata,indent=2))
    for name in args.datasets:
        path=args.data/(name+'.csv');frame=pd.read_csv(path)
        target=args.output/(name+'_gpt2.npz')
        if target.exists():
            cached=np.load(target)
            if str(cached['data_sha256'])!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Existing LM cache belongs to different input data.')
            print('Already exists',target,flush=True);continue
        texts=frame.text.tolist()
        if args.canonicalize:texts=[re.sub(r'\s+',' ',normalize(t)).strip() for t in texts]
        signature=hashlib.sha256(path.read_bytes()).hexdigest()+revision+str(args.max_tokens)+str(args.canonicalize)
        features=extract(texts,model,tokenizer,device,args.batch_size,args.max_tokens,
                         args.output/(name+'.partial.npz'),signature)
        np.savez_compressed(target,features=features,ids=frame.id.to_numpy(),
                            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print('Saved',target,features.shape,flush=True)


if __name__=='__main__':main()
