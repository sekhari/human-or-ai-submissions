"""Exact probability recipes and frozen logistic-stack formula."""
import numpy as np
from scipy.special import expit,logit
from common import ROOT,read_json
RECIPES=['best','stack','t5-qwen','equal','midpoint','roberta']
def families(predictions):
    rb=np.mean([predictions[k] for k in ['r42','r43','r44']],axis=0)
    rl=np.mean([predictions[k] for k in ['rl1','rl2']],axis=0)
    r=expit(.5*logit(np.clip(rb,1e-12,1-1e-12))+.5*logit(np.clip(rl,1e-12,1-1e-12)))
    return np.column_stack([r,(predictions['t42']+predictions['t43'])/2,(predictions['q42']+predictions['q43'])/2])
def combine(c,recipe,stack_config=None):
    best=c@np.array([.2,.4,.4])
    if recipe=='best':return best
    if recipe=='t5-qwen':return c@np.array([0.,.5,.5])
    if recipe=='equal':return c.mean(axis=1)
    if recipe=='roberta':return c[:,0]
    if recipe not in ['stack','midpoint']:raise ValueError(recipe)
    s=stack_config or read_json(ROOT/'configs/frozen_stack.json');eps=s.get('clip',1e-6)
    stack=expit(s['intercept']+logit(np.clip(c,eps,1-eps))@np.asarray(s['logit_coefficients_R_T_Q']))
    return stack if recipe=='stack' else .5*best+.5*stack
