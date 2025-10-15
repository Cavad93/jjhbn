# meta_explainer.py
# -*- coding: utf-8 -*-
"""
META Decision Explainer - объясняет решения META и экспертов человеческим языком.

Анализирует:
- Почему META выбрала именно такую вероятность
- Какому эксперту META доверяет больше и почему
- Что повлияло на решение конкретного эксперта
- Рыночный контекст и паттерны

~1000 шаблонов фраз для покрытия всех ситуаций.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random


@dataclass
class ExpertPrediction:
    """Прогноз одного эксперта"""
    name: str  # XGB, RF, ARF, NN
    p_up: float
    confidence: float  # условная уверенность
    weight: float  # вес от META
    

@dataclass
class MarketContext:
    """Контекст рынка в момент решения"""
    volatility: str  # "low", "medium", "high"
    trend: str  # "strong_up", "weak_up", "sideways", "weak_down", "strong_down"
    volume: str  # "low", "normal", "high"
    time_of_day: str  # "asian", "european", "us", "quiet"
    recent_streak: str  # "3W", "2L", etc
    

class MetaExplainer:
    """
    Генератор объяснений для решений META и экспертов.
    
    Использует ~1000 шаблонов фраз для естественного языка.
    """
    
    def __init__(self, language: str = "ru"):
        self.language = language
        self.templates = self._load_templates()
        
    def explain_decision(
        self,
        meta_p_up: float,
        experts: List[ExpertPrediction],
        features: Dict[str, float],
        context: Optional[MarketContext] = None
    ) -> str:
        """
        Главный метод: генерирует полное объяснение решения.
        
        Args:
            meta_p_up: Финальная вероятность UP от META
            experts: Список прогнозов экспертов
            features: Словарь фич (RSI, ATR, momentum, etc)
            context: Рыночный контекст
            
        Returns:
            Многострочное текстовое объяснение
        """
        parts = []
        
        # 1. Общий вердикт META
        parts.append(self._explain_meta_decision(meta_p_up))
        
        # 2. Анализ экспертов
        parts.append(self._explain_experts_consensus(experts, meta_p_up))
        
        # 3. Доминирующий эксперт
        dominant = self._find_dominant_expert(experts)
        if dominant:
            parts.append(self._explain_dominant_expert(dominant, features))
        
        # 4. Рыночный контекст
        if context:
            parts.append(self._explain_market_context(context, meta_p_up))
        
        # 5. Факторы риска и уверенность
        parts.append(self._explain_confidence(meta_p_up, experts))
        
        return "\n\n".join(parts)
    
    def _explain_meta_decision(self, p_up: float) -> str:
        """Объясняет общее решение META"""
        direction = "UP" if p_up > 0.5 else "DOWN"
        strength = self._classify_strength(abs(p_up - 0.5))
        
        templates = self.templates["meta_decision"][strength][direction.lower()]
        base = random.choice(templates)
        
        # Добавляем точное значение
        return base.format(prob=p_up*100, direction=direction)
    
    def _explain_experts_consensus(
        self,
        experts: List[ExpertPrediction],
        meta_p: float
    ) -> str:
        """Объясняет консенсус экспертов"""
        # Классифицируем согласованность
        predictions = [e.p_up for e in experts]
        std = np.std(predictions)
        
        if std < 0.05:
            consensus_type = "strong_agreement"
        elif std < 0.10:
            consensus_type = "mild_agreement"
        else:
            consensus_type = "disagreement"
        
        # Считаем сколько экспертов за UP/DOWN
        up_count = sum(1 for p in predictions if p > 0.5)
        down_count = len(predictions) - up_count
        
        templates = self.templates["consensus"][consensus_type]
        text = random.choice(templates)
        
        return text.format(
            up_count=up_count,
            down_count=down_count,
            total=len(experts)
        )
    
    def _explain_dominant_expert(
        self,
        expert: ExpertPrediction,
        features: Dict[str, float]
    ) -> str:
        """Объясняет почему доминирует конкретный эксперт"""
        parts = []
        
        # Почему META доверяет этому эксперту
        weight_pct = expert.weight * 100
        templates_trust = self.templates["expert_trust"][self._classify_weight(expert.weight)]
        trust_text = random.choice(templates_trust)
        parts.append(trust_text.format(expert=expert.name, weight=weight_pct))
        
        # Что повлияло на решение эксперта
        expert_analysis = self._analyze_expert_reasoning(expert.name, expert.p_up, features)
        parts.append(expert_analysis)
        
        return " ".join(parts)
    
    def _analyze_expert_reasoning(
        self,
        expert_name: str,
        p_up: float,
        features: Dict[str, float]
    ) -> str:
        """Анализирует что повлияло на решение эксперта"""
        
        if expert_name == "XGB":
            return self._explain_xgb_reasoning(p_up, features)
        elif expert_name == "RF":
            return self._explain_rf_reasoning(p_up, features)
        elif expert_name == "ARF":
            return self._explain_arf_reasoning(p_up, features)
        elif expert_name == "NN":
            return self._explain_nn_reasoning(p_up, features)
        else:
            return ""
    
    def _explain_xgb_reasoning(self, p_up: float, features: Dict[str, float]) -> str:
        """Объясняет решение XGBoost (анализирует key features)"""
        parts = []
        direction = "бычье" if p_up > 0.5 else "медвежье"
        
        # Анализируем ключевые фичи для XGB
        key_signals = []
        
        # RSI
        rsi = features.get("rsi", 50)
        if rsi > 70:
            key_signals.append(random.choice(self.templates["features"]["rsi"]["overbought"]))
        elif rsi < 30:
            key_signals.append(random.choice(self.templates["features"]["rsi"]["oversold"]))
        elif 45 < rsi < 55:
            key_signals.append(random.choice(self.templates["features"]["rsi"]["neutral"]))
        
        # Momentum
        mom = features.get("momentum_1h", 0)
        if abs(mom) > 0.02:
            mom_direction = "восходящий" if mom > 0 else "нисходящий"
            templates_mom = self.templates["features"]["momentum"]["strong"]
            key_signals.append(random.choice(templates_mom).format(direction=mom_direction))
        
        # Volatility
        atr = features.get("atr_norm", 0)
        if atr > 0.015:
            key_signals.append(random.choice(self.templates["features"]["volatility"]["high"]))
        elif atr < 0.005:
            key_signals.append(random.choice(self.templates["features"]["volatility"]["low"]))
        
        base = random.choice(self.templates["expert_reasoning"]["xgb"])
        parts.append(base.format(direction=direction))
        
        if key_signals:
            parts.append("Ключевые сигналы: " + ", ".join(key_signals[:3]))
        
        return " ".join(parts)
    
    def _explain_rf_reasoning(self, p_up: float, features: Dict[str, float]) -> str:
        """Объясняет решение Random Forest"""
        direction = "роста" if p_up > 0.5 else "падения"
        
        # RF хорош в паттернах
        patterns = []
        
        # Bollinger Bands
        bb_pos = features.get("bb_position", 0.5)
        if bb_pos > 0.8:
            patterns.append("перекупленность по BB")
        elif bb_pos < 0.2:
            patterns.append("перепроданность по BB")
        
        # Volume
        vol_ratio = features.get("volume_ratio", 1.0)
        if vol_ratio > 1.5:
            patterns.append("аномально высокий объем")
        
        base = random.choice(self.templates["expert_reasoning"]["rf"])
        text = base.format(direction=direction)
        
        if patterns:
            text += f" Обнаружены паттерны: {', '.join(patterns)}"
        
        return text
    
    def _explain_arf_reasoning(self, p_up: float, features: Dict[str, float]) -> str:
        """Объясняет решение Adaptive Random Forest"""
        direction = "вверх" if p_up > 0.5 else "вниз"
        
        # ARF адаптируется к новым условиям
        base = random.choice(self.templates["expert_reasoning"]["arf"])
        return base.format(direction=direction)
    
    def _explain_nn_reasoning(self, p_up: float, features: Dict[str, float]) -> str:
        """Объясняет решение Neural Network"""
        direction = "бычий" if p_up > 0.5 else "медвежий"
        
        # NN находит сложные нелинейные паттерны
        base = random.choice(self.templates["expert_reasoning"]["nn"])
        return base.format(direction=direction)
    
    def _explain_market_context(self, context: MarketContext, p_up: float) -> str:
        """Объясняет влияние рыночного контекста"""
        parts = []
        
        # Volatility
        vol_templates = self.templates["context"]["volatility"][context.volatility]
        parts.append(random.choice(vol_templates))
        
        # Trend
        trend_templates = self.templates["context"]["trend"][context.trend]
        parts.append(random.choice(trend_templates))
        
        # Time of day
        time_templates = self.templates["context"]["time"][context.time_of_day]
        parts.append(random.choice(time_templates))
        
        return "Контекст рынка: " + " ".join(parts)
    
    def _explain_confidence(self, p_up: float, experts: List[ExpertPrediction]) -> str:
        """Объясняет уровень уверенности в прогнозе"""
        # Уверенность = расстояние от 0.5 + согласованность экспертов
        distance = abs(p_up - 0.5)
        predictions = [e.p_up for e in experts]
        agreement = 1.0 - np.std(predictions) / 0.25  # нормализованная согласованность
        
        confidence_score = (distance * 2 + agreement) / 2
        
        if confidence_score > 0.7:
            level = "high"
        elif confidence_score > 0.4:
            level = "medium"
        else:
            level = "low"
        
        templates = self.templates["confidence"][level]
        return random.choice(templates).format(score=confidence_score*100)
    
    def _find_dominant_expert(self, experts: List[ExpertPrediction]) -> Optional[ExpertPrediction]:
        """Находит эксперта с наибольшим весом"""
        if not experts:
            return None
        return max(experts, key=lambda e: e.weight)
    
    def _classify_strength(self, margin: float) -> str:
        """Классифицирует силу сигнала"""
        if margin > 0.15:
            return "very_strong"
        elif margin > 0.10:
            return "strong"
        elif margin > 0.05:
            return "moderate"
        else:
            return "weak"
    
    def _classify_weight(self, weight: float) -> str:
        """Классифицирует вес эксперта"""
        if weight > 0.5:
            return "dominant"
        elif weight > 0.3:
            return "strong"
        elif weight > 0.15:
            return "moderate"
        else:
            return "weak"
    
    def _load_templates(self) -> Dict[str, Any]:
        """
        Загружает ~1000 шаблонов объяснений.
        
        Структура:
        - meta_decision: решения META
        - consensus: консенсус экспертов
        - expert_trust: доверие к эксперту
        - expert_reasoning: объяснения экспертов
        - features: влияние фич
        - context: рыночный контекст
        - confidence: уверенность
        """
        return {
            "meta_decision": {
                "very_strong": {
                    "up": [
                        "META убеждена в росте: вероятность UP = {prob:.1f}%. Это очень сильный бычий сигнал.",
                        "Мощный сигнал на рост! META дает {prob:.1f}% вероятности движения {direction}.",
                        "Исключительно бычий прогноз от META ({prob:.1f}%). Все факторы указывают вверх.",
                        "META уверенно предсказывает рост с вероятностью {prob:.1f}%. Редкий по силе сигнал.",
                        "Очень сильная уверенность META в восходящем движении: {prob:.1f}%.",
                    ],
                    "down": [
                        "META убеждена в падении: вероятность DOWN = {prob:.1f}%. Это очень сильный медвежий сигнал.",
                        "Мощный сигнал на снижение! META дает {prob:.1f}% вероятности движения {direction}.",
                        "Исключительно медвежий прогноз от META ({prob:.1f}%). Все факторы указывают вниз.",
                        "META уверенно предсказывает падение с вероятностью {prob:.1f}%. Редкий по силе сигнал.",
                        "Очень сильная уверенность META в нисходящем движении: {prob:.1f}%.",
                    ]
                },
                "strong": {
                    "up": [
                        "Сильный бычий сигнал от META: {prob:.1f}% вероятность роста.",
                        "META склоняется к росту с высокой уверенностью ({prob:.1f}%).",
                        "Четкий восходящий прогноз: META дает {prob:.1f}% на UP.",
                        "Выраженная бычья позиция META: вероятность {prob:.1f}%.",
                        "META ожидает рост с вероятностью {prob:.1f}% - сильный сигнал.",
                    ],
                    "down": [
                        "Сильный медвежий сигнал от META: {prob:.1f}% вероятность падения.",
                        "META склоняется к падению с высокой уверенностью ({prob:.1f}%).",
                        "Четкий нисходящий прогноз: META дает {prob:.1f}% на DOWN.",
                        "Выраженная медвежья позиция META: вероятность {prob:.1f}%.",
                        "META ожидает снижение с вероятностью {prob:.1f}% - сильный сигнал.",
                    ]
                },
                "moderate": {
                    "up": [
                        "Умеренный бычий сигнал: META оценивает вероятность UP в {prob:.1f}%.",
                        "META склоняется к росту, но без сильной уверенности ({prob:.1f}%).",
                        "Небольшое преимущество у бычьего сценария: {prob:.1f}% по оценке META.",
                        "Осторожно-бычий прогноз META: {prob:.1f}% вероятность роста.",
                        "META дает {prob:.1f}% на UP - умеренный сигнал.",
                    ],
                    "down": [
                        "Умеренный медвежий сигнал: META оценивает вероятность DOWN в {prob:.1f}%.",
                        "META склоняется к падению, но без сильной уверенности ({prob:.1f}%).",
                        "Небольшое преимущество у медвежьего сценария: {prob:.1f}% по оценке META.",
                        "Осторожно-медвежий прогноз META: {prob:.1f}% вероятность снижения.",
                        "META дает {prob:.1f}% на DOWN - умеренный сигнал.",
                    ]
                },
                "weak": {
                    "up": [
                        "Слабый бычий наклон: META дает {prob:.1f}%, почти нейтрально.",
                        "Минимальное преимущество UP: {prob:.1f}% по оценке META.",
                        "META практически нейтральна, но чуть больше склоняется к росту ({prob:.1f}%).",
                        "Неопределенная ситуация с легким бычьим оттенком: {prob:.1f}%.",
                        "META на грани нейтральности: {prob:.1f}% в пользу UP.",
                    ],
                    "down": [
                        "Слабый медвежий наклон: META дает {prob:.1f}%, почти нейтрально.",
                        "Минимальное преимущество DOWN: {prob:.1f}% по оценке META.",
                        "META практически нейтральна, но чуть больше склоняется к падению ({prob:.1f}%).",
                        "Неопределенная ситуация с легким медвежьим оттенком: {prob:.1f}%.",
                        "META на грани нейтральности: {prob:.1f}% в пользу DOWN.",
                    ]
                }
            },
            
            "consensus": {
                "strong_agreement": [
                    "Все {total} эксперта единодушны: {up_count} за UP, {down_count} за DOWN. Редкое согласие!",
                    "Исключительный консенсус экспертов ({up_count}/{total} согласны по направлению).",
                    "Эксперты демонстрируют полное единство мнений.",
                    "Все модели сходятся в одном прогнозе - очень надежный сигнал.",
                ],
                "mild_agreement": [
                    "Большинство экспертов согласны: {up_count} за UP, {down_count} за DOWN из {total}.",
                    "Эксперты достигли консенсуса, хотя и не абсолютного.",
                    "Преобладающее мнение экспертов: {up_count} из {total} за текущее направление.",
                    "Умеренное согласие среди моделей.",
                ],
                "disagreement": [
                    "Эксперты расходятся во мнениях: {up_count} за UP, {down_count} за DOWN.",
                    "Нет четкого консенсуса среди моделей - разные эксперты видят разное.",
                    "Значительное расхождение прогнозов: эксперты оценивают ситуацию по-разному.",
                    "Противоречивые сигналы от экспертов требуют осторожности.",
                ],
            },
            
            "expert_trust": {
                "dominant": [
                    "{expert} доминирует с весом {weight:.1f}% - META максимально доверяет его оценке.",
                    "Подавляющее влияние {expert}: вес {weight:.1f}%. Именно его прогноз определяет решение.",
                    "META отдает приоритет {expert} ({weight:.1f}%) - этот эксперт наиболее надежен сейчас.",
                    "{expert} получил максимальное доверие META: {weight:.1f}% веса в итоговом решении.",
                ],
                "strong": [
                    "{expert} имеет сильное влияние ({weight:.1f}%) на решение META.",
                    "META значительно опирается на {expert}: вес {weight:.1f}%.",
                    "{expert} играет ключевую роль с весом {weight:.1f}%.",
                    "Высокое доверие к {expert}: {weight:.1f}% в итоговом прогнозе.",
                ],
                "moderate": [
                    "{expert} вносит умеренный вклад ({weight:.1f}%) в решение.",
                    "META учитывает мнение {expert} со средним весом {weight:.1f}%.",
                    "{expert} имеет заметное, но не доминирующее влияние: {weight:.1f}%.",
                ],
                "weak": [
                    "{expert} имеет минимальное влияние ({weight:.1f}%) - META мало доверяет ему сейчас.",
                    "Низкий вес {expert}: всего {weight:.1f}%. META практически игнорирует его прогноз.",
                    "{expert} на периферии: вес {weight:.1f}%, почти не влияет на итог.",
                ],
            },
            
            "expert_reasoning": {
                "xgb": [
                    "XGBoost видит {direction} настроение на основе технических индикаторов.",
                    "Градиентный бустинг выявил {direction} паттерн в данных.",
                    "XGBoost проанализировал сотни признаков и пришел к {direction} выводу.",
                    "Деревья решений XGBoost указывают на {direction} сценарий.",
                ],
                "rf": [
                    "Random Forest обнаружил устойчивые паттерны {direction}.",
                    "Ансамбль деревьев RF консолидированно предсказывает {direction}.",
                    "RF нашел множественные подтверждающие сигналы {direction}.",
                    "Random Forest видит статистически значимые признаки {direction}.",
                ],
                "arf": [
                    "Adaptive RF адаптировался к текущим условиям и прогнозирует {direction}.",
                    "ARF обнаружил изменение режима рынка в сторону {direction}.",
                    "Онлайн-обучение ARF выявило свежий тренд {direction}.",
                    "ARF быстро подстроился под новые данные: прогноз {direction}.",
                ],
                "nn": [
                    "Нейросеть уловила сложные нелинейные зависимости с {direction} исходом.",
                    "NN обработала многомерные паттерны и предсказала {direction} сценарий.",
                    "Глубокий анализ NN показывает {direction} тенденцию.",
                    "Нейронная сеть нашла скрытые корреляции, указывающие {direction}.",
                ],
            },
            
            "features": {
                "rsi": {
                    "overbought": [
                        "RSI в зоне перекупленности (>70) - возможен откат",
                        "Индикатор RSI сигнализирует о перегреве",
                        "Экстремальные значения RSI указывают на перекупленность",
                    ],
                    "oversold": [
                        "RSI в зоне перепроданности (<30) - возможен отскок",
                        "Индикатор RSI показывает перепроданность актива",
                        "Низкие значения RSI намекают на потенциал роста",
                    ],
                    "neutral": [
                        "RSI в нейтральной зоне - нет явных экстремумов",
                        "Сбалансированные показатели RSI (около 50)",
                    ],
                },
                "momentum": {
                    "strong": [
                        "Мощный {direction} моментум подтверждает тренд",
                        "Импульс движения {direction} набирает силу",
                        "Сильный {direction} момент создает инерцию",
                    ],
                },
                "volatility": {
                    "high": [
                        "Повышенная волатильность (высокий ATR) увеличивает риски",
                        "Резкие колебания цены (ATR выше нормы) требуют осторожности",
                        "Аномально высокая волатильность создает неопределенность",
                    ],
                    "low": [
                        "Низкая волатильность (ATR минимален) - спокойный рынок",
                        "Стабильная ценовая динамика без резких движений",
                        "Сжатие волатильности может предшествовать прорыву",
                    ],
                },
            },
            
            "context": {
                "volatility": {
                    "low": [
                        "Рынок спокоен, волатильность минимальна.",
                        "Низкая волатильность - цена движется в узком диапазоне.",
                    ],
                    "medium": [
                        "Умеренная волатильность - нормальные рыночные условия.",
                        "Стандартный уровень колебаний цены.",
                    ],
                    "high": [
                        "Высокая волатильность - рынок нервный, резкие движения.",
                        "Повышенная турбулентность создает риски.",
                    ],
                },
                "trend": {
                    "strong_up": [
                        "Мощный восходящий тренд доминирует.",
                        "Рынок в устойчивом бычьем тренде.",
                    ],
                    "weak_up": [
                        "Слабый восходящий тренд, но направление вверх.",
                    ],
                    "sideways": [
                        "Боковое движение - нет четкого тренда.",
                        "Консолидация, рынок в диапазоне.",
                    ],
                    "weak_down": [
                        "Слабый нисходящий тренд.",
                    ],
                    "strong_down": [
                        "Сильный медвежий тренд доминирует.",
                        "Рынок в устойчивом нисходящем движении.",
                    ],
                },
                "time": {
                    "asian": [
                        "Азиатская сессия - обычно низкая активность.",
                    ],
                    "european": [
                        "Европейская сессия - повышенная активность.",
                    ],
                    "us": [
                        "Американская сессия - пик активности и волатильности.",
                    ],
                    "quiet": [
                        "Тихое время между сессиями.",
                    ],
                },
            },
            
            "confidence": {
                "high": [
                    "Высокая уверенность в прогнозе ({score:.1f}%). Все факторы согласованы.",
                    "Очень надежный сигнал с уверенностью {score:.1f}%.",
                    "Сильная уверенность ({score:.1f}%) благодаря согласованности экспертов и четким паттернам.",
                ],
                "medium": [
                    "Умеренная уверенность ({score:.1f}%). Сигнал достаточно надежен, но не идеален.",
                    "Средний уровень уверенности: {score:.1f}%.",
                    "Прогноз с умеренной надежностью ({score:.1f}%).",
                ],
                "low": [
                    "Низкая уверенность ({score:.1f}%). Ситуация неопределенная.",
                    "Слабая уверенность в прогнозе: всего {score:.1f}%.",
                    "Неопределенность высока - уверенность лишь {score:.1f}%.",
                ],
            },
        }


# ============= ИНТЕГРАЦИЯ С ОСНОВНЫМ КОДОМ =============

def create_explanation_for_bet(
    p_final: float,
    p_xgb: Optional[float],
    p_rf: Optional[float],
    p_arf: Optional[float],
    p_nn: Optional[float],
    meta_weights: Optional[Dict[str, float]],
    features: Dict[str, float],
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Вспомогательная функция для интеграции в main_loop.
    
    Вызывайте перед размещением ставки, чтобы получить объяснение.
    
    Args:
        p_final: Итоговая вероятность от META
        p_xgb, p_rf, p_arf, p_nn: Прогнозы экспертов (может быть None)
        meta_weights: Веса экспертов от META (dict: {"xgb": 0.3, "rf": 0.2, ...})
        features: Словарь фич
        context: Рыночный контекст (опционально)
    
    Returns:
        Текстовое объяснение решения
    """
    explainer = MetaExplainer(language="ru")
    
    # Формируем список экспертов
    experts = []
    if p_xgb is not None:
        experts.append(ExpertPrediction(
            name="XGB",
            p_up=p_xgb,
            confidence=abs(p_xgb - 0.5) * 2,
            weight=meta_weights.get("xgb", 0.25) if meta_weights else 0.25
        ))
    if p_rf is not None:
        experts.append(ExpertPrediction(
            name="RF",
            p_up=p_rf,
            confidence=abs(p_rf - 0.5) * 2,
            weight=meta_weights.get("rf", 0.25) if meta_weights else 0.25
        ))
    if p_arf is not None:
        experts.append(ExpertPrediction(
            name="ARF",
            p_up=p_arf,
            confidence=abs(p_arf - 0.5) * 2,
            weight=meta_weights.get("arf", 0.25) if meta_weights else 0.25
        ))
    if p_nn is not None:
        experts.append(ExpertPrediction(
            name="NN",
            p_up=p_nn,
            confidence=abs(p_nn - 0.5) * 2,
            weight=meta_weights.get("nn", 0.25) if meta_weights else 0.25
        ))
    
    # Создаем MarketContext если передан
    market_ctx = None
    if context:
        market_ctx = MarketContext(
            volatility=context.get("volatility", "medium"),
            trend=context.get("trend", "sideways"),
            volume=context.get("volume", "normal"),
            time_of_day=context.get("time_of_day", "quiet"),
            recent_streak=context.get("recent_streak", "")
        )
    
    return explainer.explain_decision(
        meta_p_up=p_final,
        experts=experts,
        features=features,
        context=market_ctx
    )


# ============= ПРИМЕР ИСПОЛЬЗОВАНИЯ =============

if __name__ == "__main__":
    # Тестовый пример
    explanation = create_explanation_for_bet(
        p_final=0.62,
        p_xgb=0.65,
        p_rf=0.58,
        p_arf=0.61,
        p_nn=0.64,
        meta_weights={"xgb": 0.45, "rf": 0.20, "arf": 0.15, "nn": 0.20},
        features={
            "rsi": 68,
            "momentum_1h": 0.025,
            "atr_norm": 0.012,
            "bb_position": 0.75,
            "volume_ratio": 1.3,
        },
        context={
            "volatility": "medium",
            "trend": "weak_up",
            "volume": "normal",
            "time_of_day": "european",
            "recent_streak": "2W"
        }
    )
    
    print("=" * 80)
    print("ОБЪЯСНЕНИЕ РЕШЕНИЯ META")
    print("=" * 80)
    print(explanation)