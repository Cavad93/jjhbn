# metrics/calibration.py
"""
Метрики качества калибровки вероятностей

Этот модуль содержит стандартные метрики для оценки качества
калибровки вероятностных предсказаний:
- Brier Score (MSE вероятностей)
- NLL (Negative Log-Likelihood / Log Loss)
- ECE (Expected Calibration Error)
- Reliability Curve (для визуализации)
- Bin Stats (полная статистика за один проход)
"""
import numpy as np
from typing import Tuple


def brier(y, p, w=None) -> float:
    """
    Brier Score - средняя квадратичная ошибка вероятностей
    
    Args:
        y: истинные метки (0 или 1)
        p: предсказанные вероятности [0, 1]
        w: веса примеров (опционально)
    
    Returns:
        Brier Score (меньше = лучше, идеал = 0)
    
    Example:
        >>> brier([1, 0, 1], [0.9, 0.1, 0.8])
        0.03
    """
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    if w is None:
        w = np.ones_like(y)
    return float(np.average((p - y)**2, weights=w))


def nll(y, p, w=None) -> float:
    """
    Negative Log-Likelihood (Log Loss)
    
    Args:
        y: истинные метки (0 или 1)
        p: предсказанные вероятности [0, 1]
        w: веса примеров (опционально)
    
    Returns:
        NLL (меньше = лучше, идеал = 0)
    
    Example:
        >>> nll([1, 0, 1], [0.9, 0.1, 0.8])
        0.105
    """
    y = np.asarray(y).ravel().astype(float)
    p = np.clip(np.asarray(p).ravel(), 1e-12, 1-1e-12)
    if w is None:
        w = np.ones_like(y)
    return float(-np.average(y*np.log(p) + (1-y)*np.log(1-p), weights=w))


def ece(y, p, n_bins=15) -> float:
    """
    Expected Calibration Error - стандартная метрика калибровки
    
    Args:
        y: истинные метки (0 или 1)
        p: предсказанные вероятности [0, 1]
        n_bins: количество бинов для разбиения
    
    Returns:
        ECE (меньше = лучше, идеал = 0)
        
    Example:
        >>> ece([1, 0, 1, 0], [0.9, 0.1, 0.7, 0.3])
        0.0
    """
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    total = len(y)
    val = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m): 
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        val += (m.sum() / total) * abs(acc - conf)
    return float(val)


def reliability_curve(y, p, n_bins=15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reliability Curve для построения calibration plots
    
    Args:
        y: истинные метки (0 или 1)
        p: предсказанные вероятности [0, 1]
        n_bins: количество бинов для разбиения
    
    Returns:
        (confidence, accuracy) - массивы длины n_bins
        
    Example:
        >>> conf, acc = reliability_curve([1, 0, 1, 0], [0.9, 0.1, 0.7, 0.3])
        >>> # conf[i] = средняя предсказанная вероятность в бине i
        >>> # acc[i] = реальная частота положительных исходов в бине i
    """
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    conf, acc = [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m): 
            conf.append(np.nan)
            acc.append(np.nan)
        else:
            conf.append(p[m].mean())
            acc.append(y[m].mean())
    return np.array(conf), np.array(acc)


def bin_stats(y, p, n_bins=15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Полная статистика по бинам калибровки (всё за один проход)
    
    Эффективнее, чем отдельные вызовы ece() + reliability_curve(),
    так как делает только один проход по данным.
    
    Args:
        y: истинные метки (0 или 1)
        p: предсказанные вероятности [0, 1]
        n_bins: количество бинов для разбиения
    
    Returns:
        edges: границы бинов (length = n_bins + 1)
        n: количество примеров в каждом бине (length = n_bins)
        acc: реальная accuracy в каждом бине (length = n_bins)
        conf: средняя confidence в каждом бине (length = n_bins)
        ece: Expected Calibration Error (float)
    
    Example:
        >>> edges, n, acc, conf, ece_val = bin_stats(y_true, y_pred, n_bins=10)
        >>> print(f"ECE: {ece_val:.4f}")
        >>> plt.bar(range(10), n)  # гистограмма распределения
    """
    y = np.asarray(y).ravel().astype(float)
    p = np.asarray(p).ravel().astype(float)
    
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges) - 1
    n = np.bincount(idx, minlength=n_bins)
    
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    
    for b in range(n_bins):
        m = (idx == b)
        if m.any():
            acc[b] = y[m].mean()
            conf[b] = p[m].mean()
        else:
            acc[b] = np.nan
            conf[b] = (edges[b] + edges[b + 1]) / 2.0
    
    # ECE по ненулевым бинам
    mask = n > 0
    ece_val = (n[mask] / n[mask].sum() * np.abs(acc[mask] - conf[mask])).sum()
    
    return edges, n, acc, conf, float(ece_val)