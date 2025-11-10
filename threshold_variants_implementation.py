#!/usr/bin/env python3
"""
Имплементация различных вариантов Adaptive Threshold для тестирования и сравнения

Варианты:
1. Текущий (current) - существующая логика с плавными корректировками
2. Percentile 90% (percentile) - чистый процентильный подход
3. Hybrid (hybrid) - процентиль + контекст
4. Dynamic Percentile (dynamic) - адаптивный процентиль
5. Dual Threshold (dual) - оптимальная граница между win/loss
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


# ============================================================================
# ВАРИАНТ 1: Текущая реализация (для reference)
# ============================================================================

def get_threshold_current(
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float,
    wr_50: float = None,
    wr_200: float = None,
    current_drawdown: float = 0.0
) -> float:
    """
    Текущая реализация из threshold_adapter.py

    Args:
        total_closed: Общее количество закрытых сделок
        recent_hour: Количество сделок за последний час
        recent_wr: Win rate за последние 100 сделок
        calib_error: Expected Calibration Error модели
        wr_50: Win rate за последние 50 сделок
        wr_200: Win rate за последние 200 сделок
        current_drawdown: Текущая просадка от пика (0-1)

    Returns:
        threshold: Адаптивный порог (0.5-0.8)
    """
    # Холодный старт
    if total_closed < 500:
        return 0.5

    # Базовый порог
    base_threshold = 0.65 if total_closed < 50 else 0.58

    # Корректировки
    delta = 0.0

    # 1. Win rate
    def smooth_step(x, x_min, x_max, y_min, y_max):
        if x <= x_min:
            return y_min
        if x >= x_max:
            return y_max
        ratio = (x - x_min) / (x_max - x_min)
        return y_min + ratio * (y_max - y_min)

    wr_delta = smooth_step(recent_wr, 0.35, 0.70, 0.08, -0.05)
    delta += wr_delta

    # 2. Calibration error
    cal_delta = smooth_step(calib_error, 0.01, 0.10, 0.0, 0.04)
    delta += cal_delta

    # 3. Частота сделок
    freq_delta = smooth_step(recent_hour, 0, 20, -0.02, 0.03)
    delta += freq_delta

    # 4. Тренд WR
    if wr_50 is not None and wr_200 is not None and total_closed >= 200:
        wr_trend_diff = wr_50 - wr_200
        trend_delta = smooth_step(wr_trend_diff, -0.10, 0.10, 0.05, -0.02)
        delta += trend_delta

    # 5. Просадка
    dd_delta = smooth_step(current_drawdown, 0.0, 0.20, 0.0, 0.06)
    delta += dd_delta

    # Ограничения
    delta = max(-0.05, min(delta, 0.10))
    threshold = base_threshold + delta
    threshold = max(0.5, min(threshold, 0.8))

    return threshold


# ============================================================================
# ВАРИАНТ 2: Чистый Percentile 90%
# ============================================================================

def get_threshold_percentile(
    all_probabilities: List[float],
    percentile: float = 90.0
) -> float:
    """
    Чистый процентильный подход (предложение пользователя)

    Args:
        all_probabilities: Все p_up за последние 4 часа (открытые + отклоненные)
        percentile: Процентиль для отсечки (default: 90)

    Returns:
        threshold: Процентиль от распределения вероятностей
    """
    if len(all_probabilities) < 30:
        # Недостаточно данных → fallback
        return 0.58

    threshold = np.percentile(all_probabilities, percentile)

    # Ограничения
    threshold = max(0.5, min(threshold, 0.8))

    return threshold


# ============================================================================
# ВАРИАНТ 3: Hybrid (Percentile + Context) - РЕКОМЕНДУЕМЫЙ
# ============================================================================

def get_threshold_hybrid(
    all_probabilities: List[float],
    opened_positions_probs: List[float],
    recent_wr: float,
    current_drawdown: float,
    total_closed: int,
    base_percentile: float = 85.0
) -> Tuple[float, Dict[str, float]]:
    """
    Гибридный подход: Percentile как база + корректировки по контексту

    Args:
        all_probabilities: Все p_up за последние 4 часа
        opened_positions_probs: p_up позиций что открылись
        recent_wr: Win rate за последние 100 сделок
        current_drawdown: Текущая просадка
        total_closed: Общее количество закрытых сделок
        base_percentile: Базовый процентиль (default: 85)

    Returns:
        threshold: Адаптивный порог
        debug_info: Информация для отладки
    """
    debug_info = {}

    # Холодный старт
    if total_closed < 50 or len(all_probabilities) < 30:
        debug_info['mode'] = 'cold_start'
        return 0.58, debug_info

    # === ШАГ 1: Базовый percentile ===
    base_threshold = np.percentile(all_probabilities, base_percentile)
    debug_info['base_percentile'] = base_percentile
    debug_info['base_threshold'] = base_threshold

    # === ШАГ 2: Корректировки по контексту ===
    delta = 0.0

    # Win Rate
    if recent_wr < 0.45:
        wr_delta = 0.05
    elif recent_wr > 0.65:
        wr_delta = -0.03
    else:
        wr_delta = 0.0
    delta += wr_delta
    debug_info['wr_delta'] = wr_delta

    # Просадка
    if current_drawdown > 0.15:
        dd_delta = 0.06
    elif current_drawdown > 0.10:
        dd_delta = 0.04
    elif current_drawdown > 0.05:
        dd_delta = 0.02
    else:
        dd_delta = 0.0
    delta += dd_delta
    debug_info['dd_delta'] = dd_delta

    # === ШАГ 3: Quality Check ===
    # Проверяем качество позиций выше базового порога
    if total_closed >= 100 and len(opened_positions_probs) >= 20:
        high_prob_count = sum(1 for p in opened_positions_probs if p > base_threshold)
        high_prob_ratio = high_prob_count / len(opened_positions_probs)

        # Если слишком много открывается (>50%), повышаем порог
        if high_prob_ratio > 0.5:
            quality_delta = 0.04
        else:
            quality_delta = 0.0

        delta += quality_delta
        debug_info['quality_delta'] = quality_delta
        debug_info['high_prob_ratio'] = high_prob_ratio

    # === ШАГ 4: Финал ===
    threshold = base_threshold + delta
    threshold = max(0.5, min(threshold, 0.8))

    debug_info['delta_total'] = delta
    debug_info['threshold_final'] = threshold

    return threshold, debug_info


# ============================================================================
# ВАРИАНТ 4: Dynamic Percentile
# ============================================================================

def get_threshold_dynamic_percentile(
    all_probabilities: List[float],
    recent_wr: float,
    current_drawdown: float,
    total_closed: int,
    base_percentile: float = 85.0
) -> Tuple[float, Dict[str, float]]:
    """
    Динамический процентиль: сам процентиль адаптируется к ситуации

    Args:
        all_probabilities: Все p_up за последние 4 часа
        recent_wr: Win rate за последние 100 сделок
        current_drawdown: Текущая просадка
        total_closed: Общее количество закрытых сделок
        base_percentile: Базовый процентиль (default: 85)

    Returns:
        threshold: Адаптивный порог
        debug_info: Информация для отладки
    """
    debug_info = {}

    # Холодный старт
    if total_closed < 50 or len(all_probabilities) < 30:
        debug_info['mode'] = 'cold_start'
        return 0.58, debug_info

    # === АДАПТАЦИЯ ПРОЦЕНТИЛЯ ===
    percentile_value = base_percentile
    debug_info['base_percentile'] = base_percentile

    # Win Rate влияет на селективность
    if recent_wr < 0.45:
        percentile_value += 5  # Более селективно (90%)
        debug_info['wr_adjustment'] = '+5 (low WR)'
    elif recent_wr > 0.65:
        percentile_value -= 5  # Менее селективно (80%)
        debug_info['wr_adjustment'] = '-5 (high WR)'
    else:
        debug_info['wr_adjustment'] = '0'

    # Просадка влияет на осторожность
    if current_drawdown > 0.15:
        percentile_value += 8  # Очень селективно (93%)
        debug_info['dd_adjustment'] = '+8 (critical DD)'
    elif current_drawdown > 0.10:
        percentile_value += 5  # Более селективно (90%)
        debug_info['dd_adjustment'] = '+5 (high DD)'
    elif current_drawdown > 0.05:
        percentile_value += 3  # Немного селективно (88%)
        debug_info['dd_adjustment'] = '+3 (medium DD)'
    else:
        debug_info['dd_adjustment'] = '0'

    # Ограничиваем [75, 95]
    percentile_value = max(75, min(percentile_value, 95))
    debug_info['percentile_final'] = percentile_value

    # Считаем threshold
    threshold = np.percentile(all_probabilities, percentile_value)
    threshold = max(0.5, min(threshold, 0.8))

    debug_info['threshold_final'] = threshold

    return threshold, debug_info


# ============================================================================
# ВАРИАНТ 5: Dual Threshold (самый продвинутый) - ЛУЧШИЙ
# ============================================================================

def get_threshold_dual(
    opened_positions_probs: List[float],
    rejected_signals_probs: List[float],
    opened_positions_results: List[bool],
    recent_wr: float,
    current_drawdown: float,
    total_closed: int
) -> Tuple[float, Dict[str, any]]:
    """
    Dual Threshold: находит оптимальную границу между win и loss

    Args:
        opened_positions_probs: p_up позиций что открылись
        rejected_signals_probs: p_up сигналов что отклонили
        opened_positions_results: True=win, False=loss для каждой позиции
        recent_wr: Win rate за последние 100 сделок
        current_drawdown: Текущая просадка
        total_closed: Общее количество закрытых сделок

    Returns:
        threshold: Оптимальный порог
        debug_info: Детальная информация для анализа
    """
    debug_info = {}

    # Холодный старт
    if total_closed < 100:
        debug_info['mode'] = 'cold_start'
        # Fallback на percentile
        all_probs = opened_positions_probs + rejected_signals_probs
        if len(all_probs) >= 30:
            threshold = np.percentile(all_probs, 85)
        else:
            threshold = 0.58
        return threshold, debug_info

    # === РАЗДЕЛЯЕМ НА WINS И LOSSES ===
    if len(opened_positions_results) < len(opened_positions_probs):
        # Недостаточно результатов → fallback
        debug_info['mode'] = 'insufficient_results'
        all_probs = opened_positions_probs + rejected_signals_probs
        threshold = np.percentile(all_probs, 85) if len(all_probs) >= 30 else 0.58
        return threshold, debug_info

    winning_probs = [p for p, result in zip(opened_positions_probs, opened_positions_results) if result]
    losing_probs = [p for p, result in zip(opened_positions_probs, opened_positions_results) if not result]

    debug_info['winning_count'] = len(winning_probs)
    debug_info['losing_count'] = len(losing_probs)

    # Проверяем достаточно ли данных
    if len(winning_probs) < 10 or len(losing_probs) < 10:
        debug_info['mode'] = 'insufficient_split'
        all_probs = opened_positions_probs + rejected_signals_probs
        threshold = np.percentile(all_probs, 85) if len(all_probs) >= 30 else 0.58
        return threshold, debug_info

    # === НАХОДИМ МЕДИАНЫ ===
    win_median = np.median(winning_probs)
    loss_median = np.median(losing_probs)

    debug_info['win_median'] = win_median
    debug_info['loss_median'] = loss_median
    debug_info['win_q25'] = np.percentile(winning_probs, 25)
    debug_info['win_q75'] = np.percentile(winning_probs, 75)
    debug_info['loss_q25'] = np.percentile(losing_probs, 25)
    debug_info['loss_q75'] = np.percentile(losing_probs, 75)

    # === ОПТИМАЛЬНЫЙ ПОРОГ ===
    # Берем точку между медианами, но ближе к win для осторожности
    # Коэффициент 0.7 = 70% пути от loss к win
    if win_median > loss_median:
        base_threshold = loss_median + 0.7 * (win_median - loss_median)
        debug_info['mode'] = 'optimal_separation'
    else:
        # Если win_median <= loss_median, модель плохо калибрована
        # Fallback на среднее значение между ними + запас
        base_threshold = (win_median + loss_median) / 2 + 0.05
        debug_info['mode'] = 'poor_separation'
        debug_info['warning'] = 'Model poorly calibrated: win_median <= loss_median'

    debug_info['base_threshold'] = base_threshold

    # === КОРРЕКТИРОВКИ ПО КОНТЕКСТУ ===
    delta = 0.0

    # Win Rate
    if recent_wr < 0.45:
        wr_delta = 0.05
    elif recent_wr > 0.65:
        wr_delta = -0.03
    else:
        wr_delta = 0.0
    delta += wr_delta
    debug_info['wr_delta'] = wr_delta

    # Просадка
    if current_drawdown > 0.15:
        dd_delta = 0.06
    elif current_drawdown > 0.10:
        dd_delta = 0.04
    else:
        dd_delta = 0.0
    delta += dd_delta
    debug_info['dd_delta'] = dd_delta

    # === ФИНАЛ ===
    threshold = base_threshold + delta
    threshold = max(0.5, min(threshold, 0.8))

    debug_info['delta_total'] = delta
    debug_info['threshold_final'] = threshold

    # === ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА ===
    # Overlap между winning и losing распределениями
    overlap_min = max(min(winning_probs), min(losing_probs))
    overlap_max = min(max(winning_probs), max(losing_probs))
    if overlap_max > overlap_min:
        overlap_ratio = (overlap_max - overlap_min) / (max(max(winning_probs), max(losing_probs)) - min(min(winning_probs), min(losing_probs)))
        debug_info['overlap_ratio'] = overlap_ratio
        if overlap_ratio > 0.5:
            debug_info['overlap_warning'] = 'High overlap between wins and losses'

    return threshold, debug_info


# ============================================================================
# УТИЛИТЫ ДЛЯ ТЕСТИРОВАНИЯ
# ============================================================================

def compare_all_methods(
    # Данные для текущего метода
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float,
    wr_50: float,
    wr_200: float,
    current_drawdown: float,

    # Данные для новых методов
    all_probabilities: List[float],
    opened_positions_probs: List[float],
    rejected_signals_probs: List[float],
    opened_positions_results: List[bool] = None
) -> Dict[str, Tuple[float, Dict]]:
    """
    Сравнивает все методы на одних и тех же данных

    Returns:
        Dict с результатами всех методов:
        {
            'current': (threshold, debug_info),
            'percentile': (threshold, debug_info),
            'hybrid': (threshold, debug_info),
            'dynamic': (threshold, debug_info),
            'dual': (threshold, debug_info)
        }
    """
    results = {}

    # Метод 1: Текущий
    threshold = get_threshold_current(
        total_closed, recent_hour, recent_wr, calib_error,
        wr_50, wr_200, current_drawdown
    )
    results['current'] = (threshold, {'method': 'current'})

    # Метод 2: Percentile
    threshold = get_threshold_percentile(all_probabilities, percentile=90.0)
    results['percentile'] = (threshold, {'method': 'percentile'})

    # Метод 3: Hybrid
    threshold, debug = get_threshold_hybrid(
        all_probabilities, opened_positions_probs,
        recent_wr, current_drawdown, total_closed
    )
    results['hybrid'] = (threshold, debug)

    # Метод 4: Dynamic
    threshold, debug = get_threshold_dynamic_percentile(
        all_probabilities, recent_wr, current_drawdown, total_closed
    )
    results['dynamic'] = (threshold, debug)

    # Метод 5: Dual
    if opened_positions_results is not None:
        threshold, debug = get_threshold_dual(
            opened_positions_probs, rejected_signals_probs,
            opened_positions_results, recent_wr, current_drawdown, total_closed
        )
        results['dual'] = (threshold, debug)

    return results


def print_comparison(results: Dict[str, Tuple[float, Dict]]):
    """Красиво печатает сравнение методов"""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ МЕТОДОВ ADAPTIVE THRESHOLD")
    print("="*80)

    for method_name, (threshold, debug) in results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Threshold: {threshold:.4f}")

        if debug:
            print(f"  Debug info:")
            for key, value in debug.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

    print("\n" + "="*80)


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Генерируем тестовые данные
    np.random.seed(42)

    # Симуляция: модель дает вероятности
    # Winning positions обычно имеют p_up > 0.60
    # Losing positions обычно имеют p_up < 0.65

    n_opened = 50
    n_rejected = 200

    # Открытые позиции (30 wins, 20 losses)
    winning_probs = np.random.normal(0.68, 0.06, 30)  # mean=0.68, std=0.06
    losing_probs = np.random.normal(0.58, 0.08, 20)   # mean=0.58, std=0.08

    opened_probs = np.concatenate([winning_probs, losing_probs])
    opened_results = [True] * 30 + [False] * 20

    # Случайно перемешиваем
    indices = np.random.permutation(len(opened_probs))
    opened_probs = opened_probs[indices].tolist()
    opened_results = [opened_results[i] for i in indices]

    # Отклоненные сигналы (обычно низкие вероятности)
    rejected_probs = np.random.normal(0.52, 0.05, n_rejected).tolist()

    # Все вероятности
    all_probs = opened_probs + rejected_probs

    # Параметры для текущего метода
    total_closed = 150
    recent_hour = 8
    recent_wr = 0.58  # 58% win rate
    calib_error = 0.04
    wr_50 = 0.56
    wr_200 = 0.60
    current_drawdown = 0.08  # 8% просадка

    # Сравниваем все методы
    results = compare_all_methods(
        total_closed=total_closed,
        recent_hour=recent_hour,
        recent_wr=recent_wr,
        calib_error=calib_error,
        wr_50=wr_50,
        wr_200=wr_200,
        current_drawdown=current_drawdown,
        all_probabilities=all_probs,
        opened_positions_probs=opened_probs,
        rejected_signals_probs=rejected_probs,
        opened_positions_results=opened_results
    )

    # Печатаем результаты
    print_comparison(results)

    # Дополнительная статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА ДАННЫХ")
    print("="*80)
    print(f"Всего вероятностей: {len(all_probs)}")
    print(f"Открыто позиций: {len(opened_probs)} (wins: 30, losses: 20)")
    print(f"Отклонено сигналов: {len(rejected_probs)}")
    print(f"Win rate: {recent_wr:.1%}")
    print(f"Просадка: {current_drawdown:.1%}")
    print(f"\nРаспределение вероятностей:")
    print(f"  All: min={min(all_probs):.3f}, median={np.median(all_probs):.3f}, max={max(all_probs):.3f}")
    print(f"  Opened: min={min(opened_probs):.3f}, median={np.median(opened_probs):.3f}, max={max(opened_probs):.3f}")
    print(f"  Rejected: min={min(rejected_probs):.3f}, median={np.median(rejected_probs):.3f}, max={max(rejected_probs):.3f}")
    print(f"  Winning: median={np.median([p for p, r in zip(opened_probs, opened_results) if r]):.3f}")
    print(f"  Losing: median={np.median([p for p, r in zip(opened_probs, opened_results) if not r]):.3f}")
    print("="*80)
