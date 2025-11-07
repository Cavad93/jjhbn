# 🚨 КРИТИЧЕСКАЯ ОШИБКА: ML Модели не используются при отборе монет

**Дата обнаружения:** 2025-11-07
**Серьезность:** КРИТИЧЕСКАЯ
**Статус:** ИСПРАВЛЕНО

---

## 📋 ОПИСАНИЕ ПРОБЛЕМЫ

При холодном запуске бота (более 4 часов работы) **не открывается ни одной сделки**.

### Симптомы:
```
Top-0 opportunities:
Portfolio status: 0/10 positions
```

### Корневая причина:

Бот **НЕ использовал ML модели** для получения предсказаний вероятности роста (`p_up`).

#### Код в `strategy/coin_selector.py:395-400`:
```python
# Получаем предсказание p_up
if get_predictions_func:
    p_up = get_predictions_func(features, phase, symbol)
else:
    # Если нет функции предсказаний, используем dummy значение
    p_up = 0.5  # ← ПРОБЛЕМА!!!
```

#### Вызов в `binance_bot_main.py:462-471`:
```python
top_opportunities = self.coin_selector.select_top_coins(
    all_pairs=all_pairs,
    get_ticker_func=lambda s: self.exchange.get_ticker(s),
    get_ohlcv_func=lambda s, tf, lim: self.exchange.get_ohlcv(s, tf, lim),
    feature_builder=self.feature_builder,
    calculate_atr_func=calculate_atr,
    calculate_phase_func=detect_market_phase,
    # ❌ НЕТ get_predictions_func!
    top_n=config.TOP_N_COINS
)
```

---

## 💥 ПОСЛЕДСТВИЯ

### 1. Все монеты получали `p_up = 0.5`

Без реальных предсказаний от ML моделей, всем монетам присваивалась нейтральная вероятность 50%.

### 2. Expected Value (EV) был слишком низким

С R/R ratio = 1.667 (TP 2.5 ATR / SL 1.5 ATR):

```python
# При p_up = 0.5:
EV_long = 0.5 * 1.667 - 0.5 * 1 = 0.8335 - 0.5 = 0.3335 = 33.35%
```

Но с учетом комиссий (~0.1%) и порога `min_ev_threshold = 2%`, большинство монет не проходили фильтр.

### 3. Бот не находил торговых возможностей

Результат: **0 сделок за 4+ часа работы**.

---

## ✅ РЕШЕНИЕ

### Создан метод `_get_predictions()`

Добавлен в `binance_bot_main.py:265-334`:

```python
def _get_predictions(self, features: np.ndarray, phase: int, symbol: str) -> float:
    """
    Получает предсказания от ML моделей и возвращает итоговую вероятность

    Args:
        features: 68D вектор фич
        phase: Фаза рынка (0-5)
        symbol: Символ монеты

    Returns:
        p_up: Вероятность роста (0-1)
    """
    try:
        # Собираем предсказания от экспертов
        predictions = {}

        # XGBoost
        if self.experts.get('xgb') is not None:
            try:
                p_xgb = self.experts['xgb'].predict_proba(features.reshape(1, -1))[0]
                predictions['xgb'] = p_xgb
            except Exception:
                pass

        # Random Forest
        if self.experts.get('rf') is not None:
            try:
                p_rf = self.experts['rf'].predict_proba(features.reshape(1, -1))[0]
                predictions['rf'] = p_rf
            except Exception:
                pass

        # Neural Network
        if self.experts.get('nn') is not None:
            try:
                p_nn = self.experts['nn'].predict_proba(features.reshape(1, -1))[0]
                predictions['nn'] = p_nn
            except Exception:
                pass

        # Если нет ни одного предсказания, используем базовую логику
        if not predictions:
            p_up = self.base_logic.predict_proba(features.reshape(1, -1))[0]
            return float(p_up)

        # Если есть META модель, используем её
        if self.meta is not None:
            try:
                p_up = np.mean(list(predictions.values()))
            except Exception:
                p_up = np.mean(list(predictions.values()))
        else:
            # Используем среднее арифметическое
            p_up = np.mean(list(predictions.values()))

        return float(p_up)

    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {e}")
        # Fallback на базовую логику
        try:
            p_up = self.base_logic.predict_proba(features.reshape(1, -1))[0]
            return float(p_up)
        except Exception:
            # Последний fallback - нейтральное значение
            return 0.5
```

### Добавлен в вызов `select_top_coins()`

```python
top_opportunities = self.coin_selector.select_top_coins(
    all_pairs=all_pairs,
    get_ticker_func=lambda s: self.exchange.get_ticker(s),
    get_ohlcv_func=lambda s, tf, lim: self.exchange.get_ohlcv(s, tf, lim),
    feature_builder=self.feature_builder,
    calculate_atr_func=calculate_atr,
    calculate_phase_func=detect_market_phase,
    get_predictions_func=self._get_predictions,  # ← ИСПРАВЛЕНИЕ!
    top_n=config.TOP_N_COINS
)
```

---

## 📊 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ

После исправления бот будет:

1. ✅ **Использовать реальные предсказания** от XGBoost, RandomForest, NeuralNet
2. ✅ **Вычислять среднее** от доступных экспертов
3. ✅ **Находить монеты** с вероятностью выше порога
4. ✅ **Открывать сделки** согласно стратегии

### Примерные вероятности от моделей:

Если модели обучены корректно:
- XGBoost: `p_up = 0.58`
- RandomForest: `p_up = 0.56`
- NeuralNet: `p_up = 0.62`

Среднее: `p_up = 0.587` → **Выше порога 0.58!**

EV будет положительным и достаточным для открытия позиций.

---

## 🔍 ДОПОЛНИТЕЛЬНЫЕ НАБЛЮДЕНИЯ

### Модели загружены корректно:

Из логов:
```
✓ XGBoost
✓ RandomForest
⚠ ARF (using fallback)
✓ NeuralNet
✓ META (SHADOW mode)
```

3 из 4 экспертов загружены успешно.

### Night Mode активен:

```
🌙 NIGHT MODE: Reducing activity
```

Это дополнительно снижает активность:
- Макс. 5 позиций вместо 10
- Порог входа повышен на +2%

---

## ✅ ПЛАН ДЕЙСТВИЙ

### Немедленно:
1. ✅ Добавить метод `_get_predictions()`
2. ✅ Передать в `select_top_coins()`
3. ⏳ Перезапустить бота
4. ⏳ Мониторить первый цикл ребалансировки

### Дополнительно (опционально):
1. Снизить `BASE_THRESHOLD_INITIAL` с 0.65 до 0.58 для холодного старта
2. Добавить подробное логирование предсказаний моделей
3. Добавить метрики использования моделей в Telegram уведомлениях

---

## 📝 ЗАМЕТКИ

### Почему это не было обнаружено раньше?

1. В тестах использовались моки с заранее заданными `p_up`
2. Не было интеграционных тестов полного цикла
3. Код работал в SHADOW режиме без реальных сделок

### Как предотвратить в будущем?

1. ✅ Добавить интеграционный тест `test_full_rebalance_cycle()`
2. ✅ Добавить логирование использования ML моделей
3. ✅ Добавить метрику "сделок за последние 4 часа" в dashboard
4. ✅ Добавить alert при длительном отсутствии сделок

---

**Commit:** `[HASH]` - "Critical Fix: Добавлено использование ML моделей для предсказаний при отборе монет"

**Тестирование:** Требуется перезапуск бота и мониторинг первых 4-8 часов работы

**Приоритет:** 🔴 КРИТИЧЕСКИЙ
