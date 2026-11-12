"""Selected punctuation-aware TF-IDF logistic and calibrated linear SVM."""
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
PATTERN=r'(?u)\b\w+\b|[^\w\s]'
def build(kind,seed=42):
    vectorizer=TfidfVectorizer(token_pattern=PATTERN,ngram_range=(1,2),min_df=2)
    if kind=='logistic':return make_pipeline(vectorizer,LogisticRegression(C=10,solver='liblinear',max_iter=2000,tol=1e-4,random_state=seed))
    if kind!='svm':raise ValueError(kind)
    raw=make_pipeline(vectorizer,LinearSVC(C=10,dual=True,loss='squared_hinge',max_iter=20000,tol=1e-4,random_state=seed))
    # Vocabulary/IDF and margin fitting happen inside each calibration fold.
    return CalibratedClassifierCV(raw,method='sigmoid',cv=StratifiedKFold(3,shuffle=True,random_state=seed+2),ensemble=False,n_jobs=1)
def fit(kind,train,out,seed=42):
    m=build(kind,seed);m.fit(train.text,train.label);joblib.dump(m,out/'model.joblib');return m
def load(out):return joblib.load(out/'model.joblib')
