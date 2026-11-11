from pathlib import Path

Path('outputs').mkdir(exist_ok=True)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import numpy as np, pandas as pd, re, time, sys, gc
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
D = pd.read_csv('data/train.csv')
text = D.text.fillna('').astype(str)
y = D.label.to_numpy()
T = time.time()

def style(t):
    words = re.findall("\\b[\\w']+\\b", t.lower())
    nw = len(words)
    lens = np.array([len(z) for z in words]) if words else np.array([0])
    sent = [s for s in re.split('[.!?]+', t) if s.strip()]
    ns = max(1, len(sent))
    wlens = [len(re.findall("\\b[\\w']+\\b", s)) for s in sent]
    wc = max(1, nw)
    chars = max(1, len(t))
    unique = len(set(words))
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    paragraphs = len([p for p in re.split('\\n\\s*\\n', t) if p.strip()])
    return [np.log1p(nw), np.log1p(chars), np.mean(lens), np.std(lens), np.mean(wlens), np.std(wlens), ns / wc, unique / wc, unique / np.sqrt(wc), sum((len(w) <= 3 for w in words)) / wc, sum((len(w) >= 8 for w in words)) / wc, (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams)), (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams)), t.count(',') / wc, t.count(';') / wc, t.count(':') / wc, t.count('?') / wc, t.count('!') / wc, t.count('(') / wc, t.count('"') / wc, t.count("'") / wc, t.count('-') / wc, t.count('\n') / wc, paragraphs / wc, sum((ch.isdigit() for ch in t)) / chars, sum((ch.isupper() for ch in t)) / chars, sum((ch.isspace() for ch in t)) / chars]
keys = ['log_words', 'log_chars', 'avg_word_len', 'std_word_len', 'avg_sentence_words', 'std_sentence_words', 'sentences_per_word', 'type_token_ratio', 'root_ttr', 'short_word_ratio', 'long_word_ratio', 'repeated_bigram_ratio', 'repeated_trigram_ratio', 'comma_ratio', 'semicolon_ratio', 'colon_ratio', 'question_ratio', 'exclamation_ratio', 'paren_ratio', 'double_quote_ratio', 'apostrophe_ratio', 'hyphen_ratio', 'newline_ratio', 'paragraph_ratio', 'digit_fraction', 'uppercase_fraction', 'whitespace_fraction']
S = np.asarray([style(x) for x in text], dtype=np.float32)
print('STYLE', time.time() - T, flush=True)
res = []
base, _ = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=42)
vcl = TfidfVectorizer(min_df=3, max_features=2500, stop_words='english', dtype=np.float32)
Z = vcl.fit_transform(text.iloc[base])
groups = MiniBatchKMeans(n_clusters=24, random_state=42, n_init=3, batch_size=256, max_iter=80).fit_predict(Z)
splits = list(GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=42).split(base, y[base], groups))
idx = next(((a, b) for a, b in splits if len(np.unique(y[base][a])) == 2 and len(np.unique(y[base][b])) == 2 and (len(b) > 500)))
tr, va = (base[idx[0]], base[idx[1]])
print('GROUP SPLIT', len(tr), len(va), 'positive rate', y[va].mean(), flush=True)
for seed in [42]:
    scalar = StandardScaler().fit(S[tr])
    st = np.clip(scalar.transform(S), -5, 5).astype(np.float32)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=26000, sublinear_tf=True, strip_accents='unicode', dtype=np.float32)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_features=35000, sublinear_tf=True, dtype=np.float32)
    Xw = word.fit_transform(text.iloc[tr])
    Vw = word.transform(text.iloc[va])
    print('WORD', seed, time.time() - T, flush=True)
    Xc = char.fit_transform(text.iloc[tr])
    Vc = char.transform(text.iloc[va])
    print('CHAR', seed, time.time() - T, flush=True)
    X = hstack([Xw, Xc], format='csr')
    V = hstack([Vw, Vc], format='csr')
    del Xw, Vw, Xc, Vc, word, char
    gc.collect()
    for weight in [0, 0.1]:
        A = X if weight == 0 else hstack([X, csr_matrix(st[tr] * weight)], format='csr')
        B = V if weight == 0 else hstack([V, csr_matrix(st[va] * weight)], format='csr')
        yy = y[tr].astype(float)
        n = len(tr)

        def obj(v):
            z = A @ v[:-1] + v[-1]
            p = expit(z)
            return (np.logaddexp(0, z).mean() - np.mean(yy * z), np.r_[A.T @ (p - yy) / n, (p - yy).mean()])
        opt = minimize(obj, np.zeros(A.shape[1] + 1), method='L-BFGS-B', jac=True, options={'maxiter': 45, 'ftol': 1e-08})
        trauc = roc_auc_score(y[tr], A @ opt.x[:-1] + opt.x[-1])
        auc = roc_auc_score(y[va], B @ opt.x[:-1] + opt.x[-1])
        row = dict(seed=seed, style_weight=weight, train_auc=trauc, validation_auc=auc, iterations=opt.nit, converged=opt.success, elapsed_s=time.time() - T)
        res.append(row)
        print('RESULT', row, flush=True)
        pd.DataFrame(res).to_csv('outputs/stylometric_group_aware_results.csv', index=False)
        if weight:
            imp = pd.DataFrame({'feature': keys, 'coefficient': opt.x[-28:-1], 'abs_coefficient': np.abs(opt.x[-28:-1])}).sort_values('abs_coefficient', ascending=False)
            imp.to_csv(f'outputs/stylometric_coefficients_group.csv', index=False)
    del X, V
    gc.collect()
print('DONE', time.time() - T, flush=True)
