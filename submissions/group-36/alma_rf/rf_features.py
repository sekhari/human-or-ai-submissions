"""Exact style transformer used by the evaluated and exported RF."""
from collections import Counter
import re
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class StyleFeatures(BaseEstimator, TransformerMixin):
    names = [
        "word_count","mean_word_length","std_word_length","type_token_ratio",
        "hapax_token_ratio","repeated_bigram_ratio","repeated_trigram_ratio",
        "sentence_count","mean_sentence_words","std_sentence_words","sentence_length_cv",
        "comma_per_word","semicolon_per_word","colon_per_word","exclamation_per_word",
        "question_per_word","dash_per_word","quote_per_word","uppercase_char_ratio",
        "digit_char_ratio","newline_per_word",
    ]
    def fit(self, X, y=None):
        return self
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.names,dtype=object)
    def transform(self, X):
        rows=[]
        for text in X:
            text=str(text)
            words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text.lower())
            n=max(len(words),1); chars=max(len(text),1)
            freq=Counter(words)
            lengths=np.array([len(w) for w in words] or [0],dtype=float)
            pieces=[s for s in re.split(r"[.!?]+",text) if re.search(r"[A-Za-z]",s)]
            slens=np.array([len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",s)) for s in pieces] or [0],dtype=float)
            def repeat_ratio(k):
                grams=list(zip(*(words[i:] for i in range(k))))
                return 1-len(set(grams))/len(grams) if grams else 0.0
            rows.append([
                len(words),lengths.mean(),lengths.std(),len(freq)/n,
                sum(v==1 for v in freq.values())/n,repeat_ratio(2),repeat_ratio(3),
                len(pieces),slens.mean(),slens.std(),slens.std()/max(slens.mean(),1),
                text.count(",")/n,text.count(";")/n,text.count(":")/n,
                text.count("!")/n,text.count("?")/n,
                sum(text.count(c) for c in ["-","–","—"])/n,
                sum(text.count(c) for c in ['"',"'","“","”"])/n,
                sum(c.isupper() for c in text)/chars,
                sum(c.isdigit() for c in text)/chars,text.count("\n")/n,
            ])
        matrix=np.asarray(rows,dtype=np.float32)
        assert np.isfinite(matrix).all()
        return sparse.csr_matrix(matrix)
