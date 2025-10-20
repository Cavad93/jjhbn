import numpy as np
from typing import Tuple

def reliability_curve(y, p, n_bins=15) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p, bins) - 1
    conf = []; acc = []
    for b in range(n_bins):
        m = idx==b
        if not np.any(m): 
            conf.append(np.nan); acc.append(np.nan)
        else:
            conf.append(p[m].mean()); acc.append(y[m].mean())
    return np.array(conf), np.array(acc)
