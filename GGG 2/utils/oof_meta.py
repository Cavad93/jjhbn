import numpy as np
from sklearn.linear_model import LogisticRegression

def time_folds(ts, n_folds=5):
    ts = np.asarray(ts)
    order = np.argsort(ts)
    folds = np.array_split(order, n_folds)
    for i in range(1, n_folds):
        train_idx = np.concatenate(folds[:i])
        valid_idx = folds[i]
        yield train_idx, valid_idx

def fit_oof_meta(base_models, X, y, ts, C=1.0):
    oof_scores = [np.zeros(len(y)) for _ in base_models]
    for tr, va in time_folds(ts, n_folds=5):
        for j, mdl in enumerate(base_models):
            mdl.fit(X[tr], y[tr])
            oof_scores[j][va] = mdl.predict_proba(X[va])[:,1]
    Z = np.column_stack(oof_scores)
    meta = LogisticRegression(C=C, max_iter=1000)
    meta.fit(Z, y)
    return meta, oof_scores
