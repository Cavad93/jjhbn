# -*- coding: utf-8 -*-
# meta_report.py — построение графиков CMA-ES/CEM и отправка в Telegram

import os, time
from typing import Optional, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

def plot_cma_like(
    iters: Sequence[int],
    best: Sequence[float],
    median: Optional[Sequence[float]] = None,
    sigma: Optional[Sequence[float]] = None,
    phase: int = 0,
    algo: str = "CMA-ES",
    out_dir: str = "meta_reports",
) -> str:
    iters = np.asarray(iters)
    best = np.asarray(best, dtype=float)
    title = f"{algo}: фаза {phase} — сходимость"
    fig, ax1 = plt.subplots(figsize=(9, 5))
    l1, = ax1.semilogy(iters, np.clip(best, 1e-12, None), linewidth=2, label="Лучшая пригодность (log)")
    lines, labels = [l1], [l1.get_label()]
    if median is not None:
        med = np.asarray(median, dtype=float)
        l2, = ax1.semilogy(iters, np.clip(med, 1e-12, None), linewidth=1.6, linestyle="--", label="Медианная пригодность (log)")
        lines.append(l2); labels.append(l2.get_label())
    ax1.set_xlabel("Итерация")
    ax1.set_ylabel("Loss (лог. шкала)")
    ax1.grid(True, which="both", alpha=0.3)

    if sigma is not None:
        ax2 = ax1.twinx()
        l3, = ax2.plot(iters, sigma, linewidth=1.6, label="Шаг σ")
        ax2.set_ylabel("Шаг σ")
        lines.append(l3); labels.append(l3.get_label())

    ax1.legend(lines, labels, loc="upper right")
    plt.title(title)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{algo.lower()}_phase{phase}_{int(time.time())}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def send_telegram_photo(token: str, chat_id: str, photo_path: str, caption: str) -> bool:
    try:
        import requests
    except Exception:
        print("[meta-report] requests не установлен — пропущена отправка в Telegram")
        return False
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=20)
    ok = (r.status_code == 200)
    if not ok:
        print("[meta-report] sendPhoto fail:", r.status_code, r.text[:200])
    return ok

def send_telegram_text(token: str, chat_id: str, text: str) -> bool:
    try:
        import requests
    except Exception:
        print("[meta-report] requests не установлен — пропущена отправка в Telegram")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    ok = (r.status_code == 200)
    if not ok:
        print("[meta-report] sendMessage fail:", r.status_code, r.text[:200])
    return ok
