# calib/selector.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from .platt import PlattCalibrator, TemperatureScaler
from .beta import BetaCalibrator
from .isotonic import IsotonicCalibrator
from .bbq import BBQCalibrator

# ← ИСПРАВЛЕНИЕ: Импортируем ВСЕ метрики из одного места
from metrics.calibration import ece, nll, brier


def _filter_valid_data(raw_scores, y, weights=None):
    """
    Фильтрует данные, удаляя NaN/Inf значения.

    Returns:
        (filtered_scores, filtered_y, filtered_weights, n_removed)
    """
    raw_scores = np.asarray(raw_scores).ravel()
    y = np.asarray(y).ravel()

    # Находим валидные индексы (не NaN и не Inf)
    valid_mask = np.isfinite(raw_scores) & (raw_scores >= 0) & (raw_scores <= 1)

    n_removed = len(raw_scores) - np.sum(valid_mask)

    if n_removed > 0:
        print(f"[CalibratorSelector] Filtered out {n_removed}/{len(raw_scores)} invalid samples")

    filtered_scores = raw_scores[valid_mask]
    filtered_y = y[valid_mask]
    filtered_weights = weights[valid_mask] if weights is not None else None

    return filtered_scores, filtered_y, filtered_weights, n_removed


def _prob_to_logit(probs):
    """
    Преобразует вероятности в логиты с защитой от экстремальных значений.
    """
    probs = np.asarray(probs).ravel()
    probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(probs_clipped / (1 - probs_clipped))


@dataclass
class CalibratorSelector:
    min_n_platt: int = 50
    min_n_iso: int = 800
    pick_by: str = "nll"

    def _candidates(self, n):
        cands = {"platt": PlattCalibrator(), "temp": TemperatureScaler(), "beta": BetaCalibrator()}
        if n >= self.min_n_iso:
            cands["isotonic"] = IsotonicCalibrator()
            cands["bbq"] = BBQCalibrator(n_bins=20)
        return cands

    def fit_pick(self, raw_scores, y, input_is_logit=False, weights=None) -> Tuple[str, Any, Dict[str, float]]:
        raw_scores = np.asarray(raw_scores).ravel()
        y = np.asarray(y).ravel()

        if len(raw_scores) == 0 or len(y) == 0:
            raise ValueError("Empty input data")

        # НОВОЕ: Фильтруем NaN/Inf значения
        filtered_scores, filtered_y, filtered_weights, n_removed = _filter_valid_data(raw_scores, y, weights)

        if len(filtered_scores) == 0:
            raise ValueError(f"All {len(raw_scores)} samples were invalid (NaN/Inf)")

        if n_removed > len(raw_scores) * 0.5:
            print(f"[CalibratorSelector] Warning: Removed {n_removed}/{len(raw_scores)} samples ({100*n_removed/len(raw_scores):.1f}%)")

        metrics = {}
        best_name, best_obj, best_val = None, None, float('inf')

        for name, Cal in self._candidates(len(filtered_y)).items():
            try:
                # НОВОЕ: TemperatureScaler ожидает логиты, остальные - вероятности
                if name == "temp":
                    # Преобразуем вероятности в логиты для TemperatureScaler
                    if input_is_logit:
                        train_data = filtered_scores
                    else:
                        train_data = _prob_to_logit(filtered_scores)
                    obj = Cal.fit(train_data, filtered_y, sample_weight=filtered_weights)
                    # TemperatureScaler.predict_proba НЕ принимает input_is_logit
                    p = obj.predict_proba(train_data)
                else:
                    # Platt и другие калибраторы
                    obj = Cal.fit(filtered_scores, filtered_y, sample_weight=filtered_weights)

                    if name == "platt":
                        p = obj.predict_proba(filtered_scores, input_is_logit=input_is_logit)
                    else:
                        p = obj.predict_proba(filtered_scores)

                cur = nll(filtered_y, p, w=filtered_weights) if self.pick_by == "nll" else brier(filtered_y, p, w=filtered_weights)
                metrics[name] = float(cur)

                if cur < best_val:
                    best_val, best_name, best_obj = cur, name, obj
            except Exception as e:
                print(f"[CalibratorSelector] Failed to fit {name}: {e}")
                continue

        if best_obj is None:
            raise RuntimeError("All calibrators failed to fit")

        # НОВОЕ: Правильно вызываем predict_proba для финальной оценки
        if best_name == "temp":
            if input_is_logit:
                eval_data = filtered_scores
            else:
                eval_data = _prob_to_logit(filtered_scores)
            p_best = best_obj.predict_proba(eval_data)
        elif best_name == "platt":
            p_best = best_obj.predict_proba(filtered_scores, input_is_logit=input_is_logit)
        else:
            p_best = best_obj.predict_proba(filtered_scores)

        metrics["ece"] = float(ece(filtered_y, p_best))
        metrics["nll"] = float(nll(filtered_y, p_best, w=filtered_weights))
        metrics["brier"] = float(brier(filtered_y, p_best, w=filtered_weights))  # ← ДОБАВЛЕНО для полноты

        return best_name, best_obj, metrics
