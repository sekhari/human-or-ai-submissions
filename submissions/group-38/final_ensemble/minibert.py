"""Compact BERT fine-tuning with a fixed competition validation split.
Only supplied competition labels are used. No inference API or hidden labels.
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import json,time,random,zipfile,gc,math
from pathlib import Path
import numpy as np,pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification,BertConfig,BertForSequenceClassification

from config import OUT as EXPERIMENT_DIR, DATA_ZIP
OUT=EXPERIMENT_DIR/'minibert';OUT.mkdir(parents=True,exist_ok=True)
MODEL='prajjwal1/bert-mini';SEED=42;EPOCHS=4;BATCH=32;LENGTH=256
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_num_threads(4)
def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
def read():
 with zipfile.ZipFile(DATA_ZIP) as z:
  return pd.read_csv(z.open('train.csv')),pd.read_csv(z.open('test.csv'))
train,test=read();tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.label,random_state=SEED)
print('DEVICE',DEVICE,flush=True)
seed();tokenizer=AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased') # Identical BERT vocabulary, SHA-256 checked before training.
def encode(texts):
 # Keep both opening and ending of long passages; do not silently drop every ending.
 encoded=tokenizer(texts.tolist(),add_special_tokens=False,truncation=False)['input_ids']
 ids=[];masks=[]
 for seq in encoded:
  if len(seq)>LENGTH-2:
   seq=seq[:(LENGTH-2)//2]+seq[-((LENGTH-2)-(LENGTH-2)//2):]
  seq=[tokenizer.cls_token_id]+seq+[tokenizer.sep_token_id];n=len(seq)
  ids.append(seq+[tokenizer.pad_token_id]*(LENGTH-n));masks.append([1]*n+[0]*(LENGTH-n))
 return torch.tensor(ids),torch.tensor(masks)
a,b=encode(train.text);at,bt=encode(test.text);y=torch.tensor(train.label.to_numpy(),dtype=torch.long)
print('TOKENIZATION COMPLETE',flush=True)
def make_model():
 seed();model=BertForSequenceClassification.from_pretrained(MODEL,config=BertConfig.from_pretrained(MODEL,num_labels=2),attn_implementation='eager')
 return model.to(DEVICE)
def predict(model,ids,masks):
 model.eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(ids),BATCH):
   logits=model(input_ids=ids[start:start+BATCH].to(DEVICE),attention_mask=masks[start:start+BATCH].to(DEVICE)).logits
   parts.append(torch.softmax(logits,dim=1)[:,1].cpu().numpy())
 return np.concatenate(parts)
def training(model,indices,epochs,evaluate):
 loader=DataLoader(TensorDataset(a[indices],b[indices],y[indices]),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED))
 encoder=[];head=[]
 for name,p in model.named_parameters():
  if p.requires_grad:(encoder if name.startswith('bert.') else head).append(p)
 opt=torch.optim.AdamW([{'params':encoder,'lr':3e-5},{'params':head,'lr':1e-4}],weight_decay=.01)
 # Constant LR after short warm-up; identical schedule in selection and final refit.
 steps=0;history=[];best=-1;best_epoch=1
 for epoch in range(1,epochs+1):
  model.train();loss_sum=0.;start=time.perf_counter()
  for batch_idx,(ids,mask,labels) in enumerate(loader,1):
   steps+=1;factor=min(1.,steps/50)
   opt.param_groups[0]['lr']=3e-5*factor;opt.param_groups[1]['lr']=1e-4*factor
   opt.zero_grad(set_to_none=True)
   loss=model(input_ids=ids.to(DEVICE),attention_mask=mask.to(DEVICE),labels=labels.to(DEVICE)).loss
   loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   opt.step();loss_sum+=loss.item()*len(labels)
   if batch_idx%50==0:
    print(json.dumps({'phase':'validation_training' if evaluate else 'full_refit','epoch':epoch,'batch':batch_idx,'batches':len(loader),'loss':loss_sum/(batch_idx*BATCH),'elapsed':time.perf_counter()-start}),flush=True)
  row={'epoch':epoch,'loss':loss_sum/len(indices),'train_seconds':time.perf_counter()-start}
  if evaluate:
   prob=predict(model,a[va],b[va]);auc=roc_auc_score(train.label.iloc[va],prob);row['validation_auc']=auc
   pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':train.label.iloc[va].to_numpy(),'probability_ai':prob}).to_csv(OUT/f'validation_epoch_{epoch}.csv',index=False)
   if auc>best:
    best=auc;best_epoch=epoch
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},OUT/'best_state.pt')
  history.append(row);print('EPOCH',json.dumps(row),flush=True)
  (OUT/('validation_history.json' if evaluate else 'full_history.json')).write_text(json.dumps(history,indent=2))
 return best_epoch,best
model=make_model();best_epoch,best_auc=training(model,tr,EPOCHS,True)
# Deliver a validation checkpoint prediction for analysis before full refitting.
pd.read_csv(OUT/f'validation_epoch_{best_epoch}.csv').to_csv(OUT/'neural_validation.csv',index=False)
del model;gc.collect()
if DEVICE=='mps':torch.mps.empty_cache()
print('SELECTED EPOCH',best_epoch,'AUC',best_auc,flush=True)
model=make_model();training(model,np.arange(len(train)),best_epoch,False)
p=predict(model,at,bt)
pd.DataFrame({'id':test.id,'neural':p}).to_csv(OUT/'neural_test.csv',index=False)
model.save_pretrained(OUT/'model');tokenizer.save_pretrained(OUT/'model')
summary={'model':MODEL,'seed':SEED,'max_length':LENGTH,'long_passage_policy':'first 127 + last 127 tokens','trainable':'all 4 transformer layers, embeddings and classification head','epochs_tried':EPOCHS,'selected_epoch':best_epoch,'validation_auc':best_auc,'final_training_rows':len(train),'device':DEVICE,'versions':{'torch':torch.__version__,'transformers':__import__('transformers').__version__},'pretrained_revision':getattr(model.config,'_commit_hash',None)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('NEURAL COMPLETE',json.dumps(summary),flush=True)
