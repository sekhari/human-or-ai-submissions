"""Character CNN trained only on supplied labels. No external weights or APIs."""
import os
os.environ['OMP_NUM_THREADS']='4'
import json,random,time,zipfile,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from config import OUT as EXPERIMENT_DIR, DATA_ZIP
OUT=EXPERIMENT_DIR/'cnn';OUT.mkdir(parents=True,exist_ok=True)
DEVICE='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(4);SEED=42;MAXLEN=1024;BATCH=64;EPOCHS=6
with zipfile.ZipFile(DATA_ZIP) as z:
 train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'))
def encode(text):
 result=np.zeros((len(text),MAXLEN),dtype=np.int64)
 for i,t in enumerate(text):
  b=t.encode('utf8')
  if len(b)>MAXLEN:b=b[:768]+b[-256:]
  result[i,:len(b)]=np.frombuffer(b,dtype=np.uint8).astype(np.int64)+1
 return torch.tensor(result)
x=encode(train.text);xt=encode(test.text);y=torch.tensor(train.label.to_numpy(),dtype=torch.float32);tr,va=train_test_split(np.arange(len(train)),test_size=.2,stratify=train.label,random_state=42)
def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42)
class CharCNN(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(257,32,padding_idx=0);self.convs=nn.ModuleList([nn.Conv1d(32,64,k,padding=k//2) for k in [3,5,7]]);self.head=nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1));self.drop=nn.Dropout(.15)
 def forward(self,ids):
  mask=(ids!=0).unsqueeze(1);h=self.drop(self.emb(ids)).transpose(1,2);pooled=[]
  for conv in self.convs:
   a=torch.relu(conv(h));pooled+=[a.masked_fill(~mask,-1e4).amax(2),(a*mask).sum(2)/mask.sum(2).clamp(min=1)]
  return self.head(torch.cat(pooled,1)).squeeze(1)
def predict(m,z):
 m.eval();p=[]
 with torch.inference_mode():
  for i in range(0,len(z),BATCH):p.append(torch.sigmoid(m(z[i:i+BATCH].to(DEVICE))).cpu().numpy())
 return np.concatenate(p)
def fit(indices,epochs,evaluate):
 seed();m=CharCNN().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);loader=DataLoader(TensorDataset(x[indices],y[indices]),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(42));best=-1;be=1;history=[]
 for ep in range(1,epochs+1):
  m.train();start=time.perf_counter();loss_sum=0
  for i,(ids,label) in enumerate(loader,1):
   opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(m(ids.to(DEVICE)),label.to(DEVICE));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();loss_sum+=loss.item()*len(label)
   if i%50==0:print('progress',evaluate,ep,i,len(loader),'seconds',round(time.perf_counter()-start,1),flush=True)
  row={'epoch':ep,'loss':loss_sum/len(indices),'seconds':time.perf_counter()-start}
  if evaluate:
   p=predict(m,x[va]);auc=roc_auc_score(y[va],p);row['validation_auc']=auc
   if auc>best:best=auc;be=ep;torch.save(m.state_dict(),OUT/'best_validation.pt');pd.DataFrame({'id':train.id.iloc[va].to_numpy(),'label':y[va].numpy(),'cnn':p}).to_csv(OUT/'validation.csv',index=False)
  history.append(row);print(json.dumps(row),flush=True);(OUT/('validation_history.json' if evaluate else 'full_history.json')).write_text(json.dumps(history,indent=2))
 return m,be,best
print('DEVICE',DEVICE,flush=True);m,ep,auc=fit(tr,EPOCHS,True);del m;gc.collect()
if DEVICE=='mps':torch.mps.empty_cache()
m,_,_=fit(np.arange(len(train)),ep,False);p=predict(m,xt);pd.DataFrame({'id':test.id,'cnn':p}).to_csv(OUT/'test.csv',index=False);torch.save(m.state_dict(),OUT/'full_model.pt');(OUT/'summary.json').write_text(json.dumps({'best_epoch':ep,'validation_auc':auc,'seed':42,'length':1024,'device':DEVICE},indent=2));print('CNN COMPLETE',flush=True)
