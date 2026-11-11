"""Four complementary methods with calibration fitted before inference."""
import numpy as np
from .pipeline import Blend
from .calibration import RankCalibration

WEIGHTS=np.array([.375,.225,.15,.25])  # reference, dual-LM features, DistilRoBERTa, DeBERTa


def short_lines(text):
    """Word-preserving stress view, identical to the validation transformation."""
    import hashlib
    import random
    rng=random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big'))
    words=text.split();lines=[];start=0
    while start<len(words):
        end=start+rng.randint(4,10);lines.append(' '.join(words[start:end]));start=end
    return '\n\n'.join('\n'.join(lines[i:i+4]) for i in range(0,len(lines),4))


class RobustEnsemble:
    def __init__(self,classical=None,calibration=None,weights=WEIGHTS):
        self.classical=classical
        self.calibration=calibration
        self.weights=np.asarray(weights,dtype=float).copy()
        if self.weights.shape!=(4,) or np.any(self.weights<0) or not np.isclose(self.weights.sum(),1):
            raise ValueError('Expected four nonnegative weights summing to one.')

    @staticmethod
    def fit_classical(texts,labels,features):
        # Reuse the committed reference and LM recipe. No encoder is fitted here.
        return Blend(seed=3737,weights=(.55,.25,0),threads=5).fit(texts,labels,features)

    def components(self,texts,features,distil,deberta):
        parts=self.classical.parts(texts,features)
        return np.column_stack([parts['reference'],parts['surprisal'],distil,deberta])

    def fit_calibration(self,texts,labels,features,distil,deberta):
        views=(texts,[' '.join(s.split()) for s in texts],[short_lines(s) for s in texts])
        raw=[self.components(view,features,distil,deberta) for view in views]
        self.calibration=RankCalibration().fit(np.concatenate(raw))
        return self

    def score(self,texts,features,distil,deberta):
        raw=self.components(texts,features,distil,deberta)
        return np.clip(self.calibration.transform(raw)@self.weights,0,1)
