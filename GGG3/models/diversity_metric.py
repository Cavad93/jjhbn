# -*- coding: utf-8 -*-
"""
diversity_metric.py — Измерение разнообразия экспертов

Цель: Мониторить насколько эксперты расходятся в предсказаниях

Метрики:
1. **Disagreement**: Среднее попарное расхождение |p_i - p_j|
2. **Entropy**: Энтропия распределения вероятностей
3. **Correlation**: Корреляция предсказаний между экспертами
4. **Std Dev**: Стандартное отклонение предсказаний

Интерпретация:
- **Low diversity** (<0.05): Эксперты слишком похожи → риск overfitting
- **Medium diversity** (0.05-0.15): Хорошее разнообразие
- **High diversity** (>0.15): Эксперты сильно расходятся → uncertainty

Alert System:
- ⚠️ LOW_DIVERSITY: Disagreement < 0.05
- ⚠️ HIGH_DISAGREEMENT: Disagreement > 0.20
- ⚠️ HIGH_CORRELATION: Correlation > 0.95

Автор: Claude (Anthropic)
Дата: 2025-11-10
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum

import numpy as np


# ========== ALERT TYPES ==========

class AlertType(Enum):
    """Типы алертов diversity metric"""
    LOW_DIVERSITY = "LOW_DIVERSITY"           # Эксперты слишком похожи
    HIGH_DISAGREEMENT = "HIGH_DISAGREEMENT"   # Эксперты сильно расходятся
    HIGH_CORRELATION = "HIGH_CORRELATION"     # Высокая корреляция
    OK = "OK"                                 # Все нормально


@dataclass
class DiversityMetrics:
    """Метрики разнообразия"""
    disagreement: float      # Среднее попарное расхождение [0, 1]
    entropy: float           # Энтропия [0, inf)
    std_dev: float           # Стандартное отклонение [0, 1]
    correlation_matrix: Dict[str, Dict[str, float]]  # Корреляционная матрица
    max_correlation: float   # Максимальная корреляция
    alert: AlertType        # Тип алерта
    alert_message: str      # Сообщение алерта

    def __str__(self) -> str:
        return (
            f"DiversityMetrics(\n"
            f"  disagreement={self.disagreement:.4f},\n"
            f"  entropy={self.entropy:.4f},\n"
            f"  std_dev={self.std_dev:.4f},\n"
            f"  max_correlation={self.max_correlation:.4f},\n"
            f"  alert={self.alert.value}\n"
            f")"
        )


# ========== DIVERSITY MONITOR ==========

class DiversityMonitor:
    """
    Мониторинг разнообразия экспертов

    Использование:
        monitor = DiversityMonitor(window_size=100)

        # Добавляем предсказания
        monitor.add_predictions(p_xgb=0.7, p_rf=0.65, p_arf=0.6, p_nn=0.68)

        # Получаем метрики
        metrics = monitor.get_metrics()
        print(metrics)

        # Проверяем алерты
        if metrics.alert != AlertType.OK:
            print(f"⚠️ {metrics.alert_message}")
    """

    def __init__(
        self,
        window_size: int = 100,
        low_diversity_threshold: float = 0.05,
        high_disagreement_threshold: float = 0.20,
        high_correlation_threshold: float = 0.95
    ):
        """
        Args:
            window_size: Размер скользящего окна для метрик
            low_diversity_threshold: Порог для LOW_DIVERSITY alert
            high_disagreement_threshold: Порог для HIGH_DISAGREEMENT alert
            high_correlation_threshold: Порог для HIGH_CORRELATION alert
        """
        self.window_size = window_size
        self.low_diversity_threshold = low_diversity_threshold
        self.high_disagreement_threshold = high_disagreement_threshold
        self.high_correlation_threshold = high_correlation_threshold

        # История предсказаний экспертов
        self.history_xgb = deque(maxlen=window_size)
        self.history_rf = deque(maxlen=window_size)
        self.history_arf = deque(maxlen=window_size)
        self.history_nn = deque(maxlen=window_size)

        # Счетчики
        self.n_samples = 0
        self.n_alerts = {alert.value: 0 for alert in AlertType}

    def add_predictions(
        self,
        p_xgb: Optional[float],
        p_rf: Optional[float],
        p_arf: Optional[float],
        p_nn: Optional[float]
    ):
        """
        Добавить предсказания экспертов

        Args:
            p_xgb, p_rf, p_arf, p_nn: Вероятности от экспертов
        """
        # Проверка и нормализация
        def _safe(p):
            if p is None or not np.isfinite(p):
                return None
            return float(np.clip(p, 0.0, 1.0))

        p_xgb = _safe(p_xgb)
        p_rf = _safe(p_rf)
        p_arf = _safe(p_arf)
        p_nn = _safe(p_nn)

        # Добавляем в историю
        self.history_xgb.append(p_xgb)
        self.history_rf.append(p_rf)
        self.history_arf.append(p_arf)
        self.history_nn.append(p_nn)

        self.n_samples += 1

    def get_metrics(self, min_samples: int = 20) -> Optional[DiversityMetrics]:
        """
        Рассчитать метрики разнообразия

        Args:
            min_samples: Минимум примеров для расчета

        Returns:
            DiversityMetrics или None если недостаточно данных
        """
        if self.n_samples < min_samples:
            return None

        # Собираем текущие данные
        predictions = {
            'xgb': list(self.history_xgb),
            'rf': list(self.history_rf),
            'arf': list(self.history_arf),
            'nn': list(self.history_nn)
        }

        # Фильтруем None
        valid_predictions = {}
        for name, preds in predictions.items():
            valid = [p for p in preds if p is not None]
            if len(valid) >= min_samples:
                valid_predictions[name] = np.array(valid)

        if len(valid_predictions) < 2:
            return None

        # 1. Disagreement (попарное расхождение)
        disagreement = self._calculate_disagreement(valid_predictions)

        # 2. Entropy
        entropy = self._calculate_entropy(valid_predictions)

        # 3. Std Dev
        std_dev = self._calculate_std_dev(valid_predictions)

        # 4. Correlation Matrix
        correlation_matrix, max_correlation = self._calculate_correlations(valid_predictions)

        # 5. Alert
        alert, alert_message = self._check_alerts(
            disagreement, max_correlation
        )

        # Обновляем счетчик алертов
        self.n_alerts[alert.value] += 1

        return DiversityMetrics(
            disagreement=disagreement,
            entropy=entropy,
            std_dev=std_dev,
            correlation_matrix=correlation_matrix,
            max_correlation=max_correlation,
            alert=alert,
            alert_message=alert_message
        )

    def _calculate_disagreement(self, predictions: Dict[str, np.ndarray]) -> float:
        """
        Среднее попарное расхождение

        Formula: mean(|p_i - p_j|) for all pairs (i, j)
        """
        names = list(predictions.keys())
        disagreements = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p_i = predictions[names[i]]
                p_j = predictions[names[j]]

                # Берем минимальную длину
                min_len = min(len(p_i), len(p_j))
                p_i = p_i[:min_len]
                p_j = p_j[:min_len]

                # Расхождение
                disagreement = np.mean(np.abs(p_i - p_j))
                disagreements.append(disagreement)

        return float(np.mean(disagreements)) if disagreements else 0.0

    def _calculate_entropy(self, predictions: Dict[str, np.ndarray]) -> float:
        """
        Энтропия распределения вероятностей

        Измеряет разнообразие через энтропию
        """
        # Усредняем предсказания по всем экспертам
        all_preds = []
        for preds in predictions.values():
            all_preds.extend(preds)

        if len(all_preds) == 0:
            return 0.0

        # Бинаризуем в гистограмму
        hist, _ = np.histogram(all_preds, bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # Нормализация

        # Энтропия
        hist = hist[hist > 0]  # Убираем нули
        entropy = -np.sum(hist * np.log(hist))

        return float(entropy)

    def _calculate_std_dev(self, predictions: Dict[str, np.ndarray]) -> float:
        """
        Стандартное отклонение между экспертами

        Для каждого примера считаем std между экспертами, затем усредняем
        """
        # Формируем матрицу [n_samples, n_experts]
        names = list(predictions.keys())
        min_len = min(len(predictions[name]) for name in names)

        matrix = np.array([predictions[name][:min_len] for name in names]).T

        # Std по экспертам для каждого примера
        std_per_sample = np.std(matrix, axis=1)

        return float(np.mean(std_per_sample))

    def _calculate_correlations(
        self,
        predictions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """
        Корреляционная матрица между экспертами

        Returns:
            (correlation_matrix, max_correlation)
        """
        names = list(predictions.keys())
        corr_matrix = {}
        max_corr = 0.0

        for i, name_i in enumerate(names):
            corr_matrix[name_i] = {}
            for j, name_j in enumerate(names):
                if i == j:
                    corr_matrix[name_i][name_j] = 1.0
                    continue

                p_i = predictions[name_i]
                p_j = predictions[name_j]

                # Берем минимальную длину
                min_len = min(len(p_i), len(p_j))
                p_i = p_i[:min_len]
                p_j = p_j[:min_len]

                # Корреляция Пирсона
                if len(p_i) > 1:
                    corr = np.corrcoef(p_i, p_j)[0, 1]
                    corr = float(corr) if np.isfinite(corr) else 0.0
                else:
                    corr = 0.0

                corr_matrix[name_i][name_j] = corr

                # Обновляем max (кроме диагонали)
                if i != j:
                    max_corr = max(max_corr, corr)

        return corr_matrix, max_corr

    def _check_alerts(
        self,
        disagreement: float,
        max_correlation: float
    ) -> Tuple[AlertType, str]:
        """
        Проверка условий для алертов

        Returns:
            (alert_type, alert_message)
        """
        # 1. LOW_DIVERSITY: эксперты слишком похожи
        if disagreement < self.low_diversity_threshold:
            return (
                AlertType.LOW_DIVERSITY,
                f"⚠️ Низкое разнообразие экспертов: disagreement={disagreement:.4f} < {self.low_diversity_threshold}"
            )

        # 2. HIGH_DISAGREEMENT: эксперты сильно расходятся
        if disagreement > self.high_disagreement_threshold:
            return (
                AlertType.HIGH_DISAGREEMENT,
                f"⚠️ Высокое расхождение экспертов: disagreement={disagreement:.4f} > {self.high_disagreement_threshold}"
            )

        # 3. HIGH_CORRELATION: высокая корреляция
        if max_correlation > self.high_correlation_threshold:
            return (
                AlertType.HIGH_CORRELATION,
                f"⚠️ Высокая корреляция экспертов: max_corr={max_correlation:.4f} > {self.high_correlation_threshold}"
            )

        # OK
        return (AlertType.OK, "✅ Разнообразие экспертов в норме")

    def get_alert_statistics(self) -> Dict[str, int]:
        """Статистика по алертам"""
        return self.n_alerts.copy()

    def reset(self):
        """Сброс истории"""
        self.history_xgb.clear()
        self.history_rf.clear()
        self.history_arf.clear()
        self.history_nn.clear()
        self.n_samples = 0


# ========== ТЕСТИРОВАНИЕ ==========

if __name__ == "__main__":
    print("="*80)
    print("ТЕСТ DIVERSITY MONITOR")
    print("="*80)

    # Создаем монитор
    print("\n1. Создание монитора...")
    monitor = DiversityMonitor(window_size=100)
    print("   ✅ Создано")

    # Сценарий 1: Низкое разнообразие (эксперты похожи)
    print("\n2. Сценарий: Низкое разнообразие...")
    np.random.seed(42)

    for i in range(50):
        # Все эксперты предсказывают почти одинаково
        p_base = 0.6 + np.random.normal(0, 0.01)
        p_xgb = p_base + np.random.normal(0, 0.01)
        p_rf = p_base + np.random.normal(0, 0.01)
        p_arf = p_base + np.random.normal(0, 0.01)
        p_nn = p_base + np.random.normal(0, 0.01)

        monitor.add_predictions(p_xgb, p_rf, p_arf, p_nn)

    metrics = monitor.get_metrics()
    print(f"   Disagreement: {metrics.disagreement:.4f}")
    print(f"   Alert: {metrics.alert.value}")
    print(f"   Message: {metrics.alert_message}")

    # Сценарий 2: Нормальное разнообразие
    print("\n3. Сценарий: Нормальное разнообразие...")
    monitor.reset()

    for i in range(50):
        # Эксперты предсказывают с умеренным разбросом
        p_base = 0.6
        p_xgb = p_base + np.random.normal(0, 0.05)
        p_rf = p_base + np.random.normal(0, 0.06)
        p_arf = p_base + np.random.normal(0, 0.07)
        p_nn = p_base + np.random.normal(0, 0.05)

        monitor.add_predictions(p_xgb, p_rf, p_arf, p_nn)

    metrics = monitor.get_metrics()
    print(f"   Disagreement: {metrics.disagreement:.4f}")
    print(f"   Std Dev: {metrics.std_dev:.4f}")
    print(f"   Entropy: {metrics.entropy:.4f}")
    print(f"   Alert: {metrics.alert.value}")
    print(f"   Message: {metrics.alert_message}")

    # Сценарий 3: Высокое расхождение
    print("\n4. Сценарий: Высокое расхождение...")
    monitor.reset()

    for i in range(50):
        # Эксперты сильно расходятся
        p_xgb = np.random.uniform(0.2, 0.4)
        p_rf = np.random.uniform(0.6, 0.8)
        p_arf = np.random.uniform(0.3, 0.5)
        p_nn = np.random.uniform(0.7, 0.9)

        monitor.add_predictions(p_xgb, p_rf, p_arf, p_nn)

    metrics = monitor.get_metrics()
    print(f"   Disagreement: {metrics.disagreement:.4f}")
    print(f"   Alert: {metrics.alert.value}")
    print(f"   Message: {metrics.alert_message}")

    # Корреляционная матрица
    print("\n5. Корреляционная матрица (нормальный случай)...")
    monitor.reset()

    for i in range(50):
        p_base = 0.5 + 0.2 * np.sin(i * 0.1)  # Волновой паттерн
        p_xgb = p_base + np.random.normal(0, 0.05)
        p_rf = p_base + np.random.normal(0, 0.06)
        p_arf = p_base + np.random.normal(0, 0.07)
        p_nn = p_base + np.random.normal(0, 0.05)

        monitor.add_predictions(p_xgb, p_rf, p_arf, p_nn)

    metrics = monitor.get_metrics()
    print("   Correlation Matrix:")
    for expert1, corrs in metrics.correlation_matrix.items():
        print(f"   {expert1:6s}: " + " ".join([f"{expert2}={corr:.3f}" for expert2, corr in corrs.items()]))

    print(f"   Max correlation: {metrics.max_correlation:.4f}")

    # Статистика
    print("\n6. Статистика алертов...")
    stats = monitor.get_alert_statistics()
    for alert_type, count in stats.items():
        print(f"   {alert_type}: {count}")

    print("\n" + "="*80)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("="*80)
