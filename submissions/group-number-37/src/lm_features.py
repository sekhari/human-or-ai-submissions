"""Frozen small/large language-model signals and cross-model differences."""
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from .surprisal import surprisal_matrix

MODELS=('gpt2','openai-community/gpt2-large')
LARGE_REVISION='32b71b12589c2f8d625668d2335a01cac3249519'


def resolve_model(model_id,offline_root=None):
    if offline_root is not None:
        path=Path(offline_root)/model_id.split('/')[-1]
        if not (path/'config.json').exists():raise ValueError(f'Offline model is missing: {path}')
        return str(path)
    if model_id==MODELS[1]:
        return snapshot_download(model_id,revision=LARGE_REVISION,token=False,
                                 allow_patterns=['*.json','merges.txt','vocab.json','model.safetensors'])
    if model_id!=MODELS[0]:raise ValueError('Unexpected language model.')
    return model_id


def combine(small,large):
    small=np.asarray(small);large=np.asarray(large)
    if small.shape!=large.shape or small.ndim!=2 or small.shape[1]!=15:
        raise ValueError('Expected matching N by 15 language-model feature matrices.')
    return np.c_[small,large,large-small]


def extract(texts,model_id,batch=8,offline_root=None,checkpoint=None):
    return surprisal_matrix(texts,model_name=resolve_model(model_id,offline_root),batch=batch,
                             log_every=100,checkpoint=checkpoint)
