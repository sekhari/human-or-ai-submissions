"""Fit calibration on held-out training rows and keep inference independent of batch composition."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ScoreCalibration:
    def fit(self, scores, labels):
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or not np.isfinite(scores).all():
            raise ValueError("Calibration requires a finite matrix of component scores.")
        labels = np.asarray(labels)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Calibration needs both human and AI examples.")
        self.scalers = []
        self.models = []
        for column in scores.T:
            scaler = StandardScaler().fit(column[:, None])
            model = LogisticRegression(C=1, random_state=3737).fit(scaler.transform(column[:, None]), labels)
            if model.coef_[0, 0] <= 0:
                raise ValueError("A component is inverted on the calibration set.")
            self.scalers.append(scaler)
            self.models.append(model)
        return self

    def transform(self, scores):
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != len(self.models) or not np.isfinite(scores).all():
            raise ValueError("Prediction columns must match calibrated components.")
        return np.column_stack([
            model.predict_proba(scaler.transform(scores[:, index, None]))[:, 1]
            for index, (scaler, model) in enumerate(zip(self.scalers, self.models))
        ])


class RankCalibration:
    """Held-out training quantiles; independent of the prediction batch."""
    def fit(self, scores, labels=None):
        from sklearn.preprocessing import QuantileTransformer
        scores=np.asarray(scores,dtype=float)
        if scores.ndim!=2 or not np.isfinite(scores).all():
            raise ValueError('Rank calibration requires a finite component matrix.')
        self.mapping=QuantileTransformer(n_quantiles=min(800,len(scores)),subsample=None,
                                         output_distribution='uniform',random_state=3737).fit(scores)
        return self

    def transform(self,scores):
        scores=np.asarray(scores,dtype=float)
        if scores.ndim!=2 or not np.isfinite(scores).all():
            raise ValueError('Prediction components must be a finite matrix.')
        return self.mapping.transform(scores)
