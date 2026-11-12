"""Additional label-free, local lexical/style features and format transforms."""
import math
import re
import string
from collections import Counter
import numpy as np
from features import normalize, style_features


def canonical(text):
    return re.sub(r'\s+',' ',normalize(text)).strip()


def reflow(text, separator='\n\n', width=65):
    words=canonical(text).split()
    return separator.join(' '.join(words[i:i+width]) for i in range(0,len(words),width))


def load_lm_matrix(artifact_dir, split, directories):
    matrices=[np.load(artifact_dir/d/(split+'_gpt2.npz'))['features'] for d in directories]
    return combine_lm_matrices(matrices)


def combine_lm_matrices(matrices):
    if len(matrices)==1:return matrices[0]
    result=list(matrices)
    for matrix in matrices[1:]:
        result.extend([matrices[0]-matrix,matrices[0]/(np.abs(matrix)+.1)])
    return np.column_stack(result).astype(np.float32)


def expanded_features(texts, canonicalize=False):
    texts=[canonical(t) if canonicalize else normalize(t) for t in texts]
    base=style_features(texts);rows=[]
    for t in texts:
        words=re.findall(r"\b\w+(?:['’-]\w+)*\b",t.lower());n=max(len(words),1)
        counts=Counter(words);freq=np.asarray(list(counts.values()) or [1],dtype=float)
        probs=freq/n;v=max(len(counts),1)
        lengths=np.array([len(w) for w in words] or [0],dtype=float)
        sentences=[s.strip() for s in re.split(r'[.!?]+',t) if s.strip()]
        sl=np.array([len(s.split()) for s in sentences] or [0],dtype=float)
        alphabetic=[w for w in words if w.isalpha()]
        char_counts=Counter(t.lower());cp=np.asarray(list(char_counts.values()) or [1])/max(len(t),1)
        entropy=-np.sum(probs*np.log(probs+1e-12))
        row=[entropy,entropy/max(np.log(n),1),np.sum(probs**2),v/np.sqrt(n),
             np.log(v)/max(np.log(n),1),10000*(np.sum(freq**2)-n)/(n*n),
             np.log(n)**2/max(np.log(n)-np.log(v),.01),n**(v**(-.165)),
             np.max(freq)/n,np.median(freq),np.std(freq),sum(freq==2)/n,
             sum(freq>=3)/n,-np.sum(cp*np.log(cp+1e-12))]
        for size in [25,50,100]:
            ratios=[len(set(words[i:i+size]))/len(words[i:i+size]) for i in range(0,max(n-size+1,1),max(size//2,1)) if words[i:i+size]] or [0]
            row.extend([np.mean(ratios),np.std(ratios),np.min(ratios),np.max(ratios)])
        for k in [2,3,4]:
            grams=Counter(tuple(words[i:i+k]) for i in range(len(words)-k+1));total=max(sum(grams.values()),1)
            row.extend([len(grams)/total,max(grams.values(),default=0)/total,
                        sum(c-1 for c in grams.values())/total])
        row.extend(np.quantile(lengths,[.1,.25,.75,.9,.95]).tolist())
        row.extend(np.quantile(sl,[.1,.25,.5,.75,.9,.95]).tolist())
        row.extend([np.mean(np.abs(np.diff(lengths))) if n>1 else 0,
            np.mean(np.abs(np.diff(sl))) if len(sl)>1 else 0,
            np.mean(sl<5),np.mean(sl>25),np.mean(sl>40),
            np.mean(lengths==1),np.mean(lengths>12),
            len(re.findall(r'\b\d+(?:\.\d+)?\b',t))/n,
            len(re.findall(r'\b[A-Z][a-z]+\b',t))/n,
            len(re.findall(r'\b[A-Z]{2,}\b',t))/n,
            len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b',t))/n,
            sum(len(re.findall(r'[aeiouy]+',w)) for w in alphabetic)/max(len(alphabetic),1),
            float(bool(t) and t[-1] in '.!?'),float(bool(t) and t[-1] in ',;:'),
            float(bool(t) and t[-1].isalnum()),float(bool(t) and t[0].islower()),
            t.count('...')/n,t.count('!')/n,t.count('?')/n])
        for ending in ['ing','ed','ly','tion','ness','ment','ity','s']:
            row.append(sum(w.endswith(ending) for w in alphabetic)/n)
        for character in string.ascii_lowercase+string.digits:
            row.append(t.lower().count(character)/max(len(t),1))
        # Paragraph/line statistics remain explicit so canonical models can remove them.
        for sep in ['\n','\n\n']:
            lines=[p for p in t.split(sep) if p.strip()]
            sizes=np.array([len(p.split()) for p in lines] or [0])
            row.extend([len(lines)/n,np.mean(sizes),np.std(sizes),np.max(sizes),np.min(sizes)])
        rows.append(row)
    result=np.column_stack([base,np.asarray(rows,dtype=np.float32)])
    return np.nan_to_num(result,nan=0,posinf=0,neginf=0).astype(np.float32)
