# -*- coding: utf-8 -*-
# expert_report.py — панель калибровки 4 экспертов из φ-датасета МЕТА
import os, time, math
import numpy as np
import matplotlib.pyplot as plt

def _sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def _bin_stats(y, p, bins=12):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges) - 1
    n = np.bincount(idx, minlength=bins)
    acc = np.zeros(bins)
    conf = np.zeros(bins)
    for b in range(bins):
        m = (idx == b)
        if m.any():
            acc[b] = y[m].mean()
            conf[b] = p[m].mean()
        else:
            acc[b] = np.nan
            conf[b] = (edges[b] + edges[b+1]) / 2.0
    # ECE по ненулевым бинам
    mask = n > 0
    ece = (n[mask] / n[mask].sum() * np.abs(acc[mask] - conf[mask])).sum()
    return edges, n, acc, conf, float(ece)

def _brier(y, p):
    return float(np.mean((p - y) ** 2))

def _wr(y, p, thr=0.5):
    return float(( (p >= thr) == (y > 0.5) ).mean())

def plot_experts_reliability_panel(X, y, phase: int, out_dir: str, bins: int = 12) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())

    # из φ берём 4 логита экспертов → вероятности
    logits = X[:, :4].astype(float)   # [logit(pxgb), logit(prf), logit(parf), logit(pnn)]
    probs  = _sigmoid(logits)
    names  = ["XGB", "RF", "ARF", "NN"]

    fig = plt.figure(figsize=(12, 8), dpi=140)
    fig.suptitle(f"Эксперты — калибровка/качество (фаза {phase})", fontsize=14)

    for j in range(4):
        pj = probs[:, j]
        ax = fig.add_subplot(2, 2, j+1)
        edges, n, acc, conf, ece = _bin_stats(y, pj, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # диагональ идеальной калибровки
        ax.plot([0,1], [0,1], linestyle="--", linewidth=1)

        # столбики «accuracy по бинам» и точки «средняя уверенность»
        ax.bar(centers, np.nan_to_num(acc), width=1.0/bins, alpha=0.35)
        ax.scatter(centers, conf, s=10)

        # метрики
        brier = _brier(y, pj)
        wr    = _wr(y, pj)
        ax.set_title(f"{names[j]} | ECE={ece:.3f} | Brier={brier:.3f} | WR={wr:.1%}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)

    path = os.path.join(out_dir, f"experts_phase{phase}_{ts}.png")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)
    return path
