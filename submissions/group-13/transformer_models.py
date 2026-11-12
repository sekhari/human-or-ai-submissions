"""Portable versions of the recorded RoBERTa, T5 and Qwen training recipes."""
import gc,math,random,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM,DataCollatorWithPadding,get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig,TaskType,get_peft_model,PeftModel
from common import ROOT,read_json,write_json
CONFIG=read_json(ROOT/'configs/models.json')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

class QwenClassifier(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.dropout=nn.Dropout(.1)
        self.head=nn.Linear(backbone.config.hidden_size,2,dtype=torch.float32)
        nn.init.normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)
    def forward(self,input_ids,attention_mask,**unused):
        h=self.backbone(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
        positions=attention_mask.sum(1)-1
        pooled=h[torch.arange(len(h),device=h.device),positions]
        return SequenceClassifierOutput(logits=self.head(self.dropout(pooled).float()))

def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else nullcontext()

def resources(spec,device,checkpoint=None,cache_dir=None,offline=False):
    seed_all(spec['seed']);kind=spec['kind'];source=CONFIG['sources'][kind]
    opts=dict(cache_dir=cache_dir,local_files_only=offline,trust_remote_code=False)
    remote=dict(**opts,revision=source['revision'])
    tokenizer=AutoTokenizer.from_pretrained(str(checkpoint/'tokenizer') if checkpoint else source['repo'],**(opts if checkpoint else remote))
    if kind=='qwen':
        tokenizer.padding_side='right'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        dtype=torch.bfloat16 if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else torch.float32
        base=AutoModel.from_pretrained(source['repo'],**remote,torch_dtype=dtype,attn_implementation='sdpa')
        base.config.use_cache=False
        if checkpoint:
            backbone=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=False,local_files_only=True)
        else:
            backbone=get_peft_model(base,LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none'))
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});backbone.enable_input_require_grads()
            for name,p in backbone.named_parameters():
                assert p.requires_grad==('lora_' in name)
                if p.requires_grad:p.data=p.data.float()
        model=QwenClassifier(backbone)
        if checkpoint:model.head.load_state_dict(torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=True))
    elif kind=='t5':
        model=AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint/'model') if checkpoint else source['repo'],**(opts if checkpoint else remote),dropout_rate=0.0)
        labels=[tokenizer.encode(t,add_special_tokens=False) for t in ['human','AI']]
        assert all(len(t)==1 for t in labels) and labels[0]!=labels[1]
        model.config.binary_label_ids=[t[0] for t in labels];model.config.use_cache=False
    else:
        extra={} if checkpoint else dict(num_labels=2,id2label={0:'human',1:'AI'},label2id={'human':0,'AI':1})
        model=AutoModelForSequenceClassification.from_pretrained(str(checkpoint/'model') if checkpoint else source['repo'],**(opts if checkpoint else remote),attn_implementation='sdpa',**extra)
        if not checkpoint and spec['gradient_checkpointing']:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model.to(device),tokenizer

def encode(tokenizer,frame,kind):
    if kind=='qwen':
        raw=tokenizer(frame.text.tolist(),add_special_tokens=False,truncation=True,max_length=511,padding=False)
        rows=[dict(input_ids=ids+[tokenizer.eos_token_id],attention_mask=[1]*(len(ids)+1)) for ids in raw['input_ids']]
    else:
        raw=tokenizer(frame.text.tolist(),truncation=True,max_length=512,padding=False)
        rows=[{k:v[i] for k,v in raw.items()} for i in range(len(frame))]
    if 'label' in frame:
        for row,y in zip(rows,frame.label):row['labels']=int(y)
    return rows

def loader(rows,tokenizer,batch_size,order=None):
    chosen=rows if order is None else [rows[int(i)] for i in order]
    return torch.utils.data.DataLoader(chosen,batch_size=batch_size,shuffle=False,num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8))

def binary_logits(model,batch,kind):
    x={k:v for k,v in batch.items() if k!='labels'}
    if kind=='t5':
        start=torch.full((len(x['input_ids']),1),model.config.decoder_start_token_id,dtype=torch.long,device=x['input_ids'].device)
        return model(**x,decoder_input_ids=start,use_cache=False).logits[:,0,model.config.binary_label_ids]
    return model(**x).logits

@torch.inference_mode()
def predict_loaded(model,tokenizer,frame,spec,device):
    model.eval();scores=[]
    for batch in loader(encode(tokenizer,frame,spec['kind']),tokenizer,spec['micro_batch']):
        batch={k:v.to(device) for k,v in batch.items()}
        with autocast(device):z=binary_logits(model,batch,spec['kind']).float()
        assert torch.isfinite(z).all()
        z=z.cpu().numpy().astype(np.float64);scores.append(z[:,1]-z[:,0])
    return expit(np.concatenate(scores))

def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def train_member(key,frame,out,device,cache_dir=None,offline=False,micro_batch=None,fold=False):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Use a new artifact directory: {out}')
    out.mkdir(parents=True);spec=dict(CONFIG['members'][key])
    if fold and spec['kind']=='roberta_large':spec['micro_batch']=4
    if micro_batch:spec['micro_batch']=micro_batch
    micro=spec['micro_batch']
    if micro<=0 or 32%micro:raise ValueError('Micro batch must be a positive divisor of 32')
    write_json(out/'spec.json',spec);frame[['id','label']].to_csv(out/'training_ids.csv',index=False)
    model,tok=resources(spec,device,cache_dir=cache_dir,offline=offline)
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim>=2],weight_decay=.01),
        dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim<2],weight_decay=0.)]
    opt=torch.optim.AdamW(groups,lr=spec['lr']);steps=math.ceil(len(frame)/32)*spec['schedule_epochs']
    scheduler=get_linear_schedule_with_warmup(opt,math.ceil(.1*steps),steps)
    rows=encode(tok,frame,spec['kind']);history=[];accum=32//micro;started=time.perf_counter()
    for epoch in range(1,spec['epochs']+1):
        model.train();opt.zero_grad(set_to_none=True);total=0.;seen=0
        order=np.random.default_rng(spec['seed']+epoch).permutation(len(frame))
        batches=loader(rows,tok,micro,order)
        for step,b in enumerate(batches,1):
            b={k:v.to(device) for k,v in b.items()};n=len(b['labels'])
            with autocast(device):loss=torch.nn.functional.cross_entropy(binary_logits(model,b,spec['kind']).float(),b['labels'])
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            group_size=min(32,len(frame)-((step-1)//accum)*32)
            (loss*n/group_size).backward();total+=float(loss.detach())*n;seen+=n
            if step%accum==0 or step==len(batches):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
                opt.step();scheduler.step();opt.zero_grad(set_to_none=True)
        row=dict(epoch=epoch,loss=total/seen,seconds=time.perf_counter()-started);history.append(row)
        write_json(out/'history.json',history);print(dict(member=key,**row),flush=True)
    before=predict_loaded(model,tok,frame.iloc[:32],spec,device)
    tok.save_pretrained(out/'tokenizer')
    if spec['kind']=='qwen':
        model.backbone.save_pretrained(out/'adapter',safe_serialization=True,save_embedding_layers=False)
        torch.save(model.head.state_dict(),out/'head.pt')
    else:model.save_pretrained(out/'model',safe_serialization=True)
    loss=None;b=None;groups=None
    del model,opt,scheduler;release()
    model,tok=resources(spec,device,checkpoint=out,cache_dir=cache_dir,offline=offline)
    after=predict_loaded(model,tok,frame.iloc[:32],spec,device);difference=float(np.max(abs(before-after)))
    assert difference<1e-5
    write_json(out/'summary.json',dict(spec=spec,source=CONFIG['sources'][spec['kind']],training_rows=len(frame),reload_max_error=difference,seconds=time.perf_counter()-started))
    del model;release()

def predict_member(folder,frame,device,cache_dir=None,offline=False):
    spec=read_json(folder/'spec.json');model,tok=resources(spec,device,checkpoint=folder,cache_dir=cache_dir,offline=offline)
    p=predict_loaded(model,tok,frame,spec,device);del model;release();return p

def train_members(frame,out,device,cache_dir=None,offline=False,micro_batch=None):
    for key in CONFIG['members']:train_member(key,frame,out/key,device,cache_dir,offline,micro_batch)

def predict_members(out,frame,device,cache_dir=None,offline=False):
    return {key:predict_member(out/key,frame,device,cache_dir,offline) for key in CONFIG['members']}
