# expert_report.py
import os, time, math
import numpy as np
import matplotlib.pyplot as plt
from metrics.calibration import bin_stats  # ← НОВЫЙ ИМПОРТ

def _sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def _brier(y, p):
    return float(np.mean((p - y) ** 2))

def _wr(y, p, thr=0.5):
    return float(( (p >= thr) == (y > 0.5) ).mean())

def plot_experts_reliability_panel(X, y, phase: int, out_dir: str, bins: int = 12) -> str:
    """
    Строит панель калибровки для 4 экспертов (XGB, RF, ARF, NN)
    
    Args:
        X: массив φ-признаков (первые 4 колонки = логиты экспертов)
        y: истинные метки
        phase: номер фазы
        out_dir: директория для сохранения графика
        bins: количество бинов для калибровки
    
    Returns:
        Путь к сохраненному файлу
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())

    # из φ берём 4 логита экспертов → вероятности
    logits = X[:, :4].astype(float)
    probs  = _sigmoid(logits)
    names  = ["XGB", "RF", "ARF", "NN"]

    fig = plt.figure(figsize=(12, 8), dpi=140)
    fig.suptitle(f"Эксперты — калибровка/качество (фаза {phase})", fontsize=14)

    for j in range(4):
        pj = probs[:, j]
        ax = fig.add_subplot(2, 2, j+1)
        
        # ← ИЗМЕНЕНИЕ: используем bin_stats из metrics.calibration
        edges, n, acc, conf, ece = bin_stats(y, pj, n_bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # диагональ идеальной калибровки
        ax.plot([0,1], [0,1], linestyle="--", linewidth=1, color='gray')

        # столбики «accuracy по бинам» и точки «средняя уверенность»
        ax.bar(centers, np.nan_to_num(acc), width=1.0/bins, alpha=0.35, color='skyblue')
        ax.scatter(centers, conf, s=10, color='red')

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
