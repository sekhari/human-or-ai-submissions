"""Enriched observable style features; labels never enter feature extraction."""
import sys,re,json,zipfile,collections,time
from pathlib import Path
import numpy as np,pandas as pd
from config import OUT, DATA_ZIP
from base_features import features as base_features
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
NLTK_DIR=OUT/'nltk_data';nltk.data.path.insert(0,str(NLTK_DIR))
nltk.download('averaged_perceptron_tagger_eng',download_dir=str(NLTK_DIR),quiet=True,raise_on_error=True)
from nltk.tag import PerceptronTagger
from nltk.tokenize import TreebankWordTokenizer
tagger=PerceptronTagger();tokenizer=TreebankWordTokenizer()
with zipfile.ZipFile(DATA_ZIP) as z:
 train=pd.read_csv(z.open('train.csv'));test=pd.read_csv(z.open('test.csv'));unlab=pd.read_csv(z.open('unlabeled.csv'))
# Fixed feature dictionary, neither fitted to validation/test text nor labels.
STOP=sorted(ENGLISH_STOP_WORDS)
POS=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','.',',',':','``',"''",'-LRB-','-RRB-']
# Universal POS bigrams reduce topical dependence.
UPOS=['N','V','J','R','P','D','I','C','M','T','NUM','PUNC','OTHER']
def coarse(t):
 if t.startswith('NN'):return 'N'
 if t.startswith('VB'):return 'V'
 if t.startswith('JJ'):return 'J'
 if t.startswith('RB'):return 'R'
 if t.startswith('PRP') or t in ['WP','WP$']:return 'P'
 if t in ['DT','PDT','WDT']:return 'D'
 if t in ['IN']:return 'I'
 if t=='CC':return 'C'
 if t=='MD':return 'M'
 if t=='TO':return 'T'
 if t=='CD':return 'NUM'
 if t in ['.',',',':','``',"''",'-LRB-','-RRB-']:return 'PUNC'
 return 'OTHER'
NAMES=[f'original_{i}' for i in range(99)]
NAMES += [f'function_{w}' for w in STOP]
NAMES += [f'wordlen_{i}' for i in range(1,21)]
NAMES += [f'letter_{c}' for c in 'abcdefghijklmnopqrstuvwxyz']
for p in ['sentence_words','sentence_chars','word_chars','paragraph_words','unique_50','unique_100','syllables']:
 NAMES += [p+'_'+k for k in ['mean','std','min','q10','q25','median','q75','q90','max','cv']]
NAMES += ['sentences_per_word','paragraphs_per_word','punctuation_per_word','vowel_ratio','yule_k','hapax','dislegomena','herdan','maas','flesch_reading_ease','flesch_kincaid','coleman_liau','ari','stop_ratio','suffix_ing','suffix_ed','suffix_ly','suffix_tion','suffix_ity','suffix_ment','suffix_ness','first_person','second_person','third_person','starting_stop','starting_upper','ending_period','double_space','tabs','word_repeat_adjacent','trigram_repeat','bigram_repeat']
NAMES += ['pos_'+p for p in POS]
NAMES += [f'pos2_{p}_{q}' for p in UPOS for q in UPOS]
NAMES += [f'stop_mask2_{i}' for i in ['00','01','10','11']]
def stats(a):
 a=np.asarray(a or [0],dtype=float);m=a.mean()
 return [m,a.std(),a.min(),*np.quantile(a,[.1,.25,.5,.75,.9]),a.max(),a.std()/(m+1e-6)]
def extract(t):
 raw=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",t);w=[v.lower() for v in raw];n=max(len(w),1);ch=max(len(t),1);c=collections.Counter(w)
 sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',t) if re.search(r'[A-Za-z]',s)] or ['']
 sentw=[len(re.findall(r'\b\w+\b',s)) for s in sentences];sentc=[len(s) for s in sentences]
 paras=[len(re.findall(r'\b\w+\b',s)) for s in re.split(r'\n\s*\n',t) if s.strip()]
 wl=[len(v) for v in w];letters=collections.Counter(t.lower());wlcount=collections.Counter(min(len(v),20) for v in w)
 windows50=[len(set(w[i:i+50]))/max(len(w[i:i+50]),1) for i in range(0,len(w),50)]
 windows100=[len(set(w[i:i+100]))/max(len(w[i:i+100]),1) for i in range(0,len(w),100)]
 syll=[max(1,len(re.findall(r'[aeiouy]+',v))-int(v.endswith('e') and not v.endswith(('le','ye')))) for v in w]
 f=base_features(t)+[c[s]/n for s in STOP]+[wlcount[i]/n for i in range(1,21)]+[letters[s]/ch for s in 'abcdefghijklmnopqrstuvwxyz']
 for a in [sentw,sentc,wl,paras,windows50,windows100,syll]:f+=stats(a)
 ns=max(len(sentences),1);v=max(len(c),1);sumfreq2=sum(z*z for z in c.values());alpha=sum(len(z) for z in w)
 f += [ns/n,len(paras)/n,sum(not z.isalnum() and not z.isspace() for z in t)/n,sum(letters[z] for z in 'aeiou')/max(alpha,1),10000*(sumfreq2-n)/n**2,sum(z==1 for z in c.values())/n,sum(z==2 for z in c.values())/n,np.log(v)/np.log(n+1),(np.log(n+1)-np.log(v))/max(np.log(n+1)**2,1),206.835-1.015*n/ns-84.6*sum(syll)/n,.39*n/ns+11.8*sum(syll)/n-15.59,.0588*(100*alpha/n)-.296*(100*ns/n)-15.8,4.71*alpha/n+.5*n/ns-21.43,sum(c[z] for z in STOP)/n]
 f += [sum(v.endswith(s) for v in w)/n for s in ['ing','ed','ly','tion','ity','ment','ness']]
 f += [sum(c[z] for z in ['i','me','my','mine','we','us','our'])/n,sum(c[z] for z in ['you','your','yours'])/n,sum(c[z] for z in ['he','him','his','she','her','they','them','their'])/n]
 f += [sum(bool(s.split()) and s.split()[0].lower() in STOP for s in sentences)/ns,sum(bool(s) and s[0].isupper() for s in sentences)/ns,sum(s.endswith('.') for s in sentences)/ns,t.count('  ')/ch,t.count('\t')/ch,sum(a==b for a,b in zip(w,w[1:]))/n]
 for k in [3,2]:
  grams=list(zip(*[w[i:] for i in range(k)]));f += [(len(grams)-len(set(grams)))/max(len(grams),1)]
 tags=[p for _,p in tagger.tag(tokenizer.tokenize(t))];tc=collections.Counter(tags);nt=max(len(tags),1)
 f += [tc[p]/nt for p in POS]
 ct=[coarse(p) for p in tags];bc=collections.Counter(zip(ct,ct[1:]));f += [bc[p,q]/max(len(ct)-1,1) for p in UPOS for q in UPOS]
 masks=['1' if v in STOP else '0' for v in w];mc=collections.Counter(a+b for a,b in zip(masks,masks[1:]));f += [mc[k]/max(n-1,1) for k in ['00','01','10','11']]
 assert len(f)==len(NAMES),(len(f),len(NAMES))
 return np.nan_to_num(f,nan=0,posinf=0,neginf=0)
if __name__=='__main__':
 start=time.perf_counter();allx=[]
 for name,frame in [('train',train),('unlabeled',unlab),('test',test)]:
  x=[]
  for i,t in enumerate(frame.text):
   x.append(extract(t))
   if (i+1)%1000==0:print(name,i+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
  allx.append(np.array(x,dtype=np.float32))
 np.savez_compressed(OUT/'enriched_features.npz',train=allx[0],unlabeled=allx[1],test=allx[2])
 (OUT/'feature_names.json').write_text(json.dumps(NAMES,indent=2));print('COMPLETE',len(NAMES),flush=True)
