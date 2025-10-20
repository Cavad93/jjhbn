import numpy as np

def ece(y, p, n_bins=15):
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(p, bins) - 1
    total = len(y)
    val = 0.0
    for b in range(n_bins):
        m = idx==b
        if not np.any(m): continue
        conf = p[m].mean()
        acc = y[m].mean()
        val += (m.sum()/total)*abs(acc-conf)
    return float(val)
