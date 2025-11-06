# 📋 ДОПОЛНЕНИЕ К BINANCE_MIGRATION_PLAN.md

**Версия:** 1.1
**Дата:** 2025-11-06
**Важные уточнения по реализации**

---

## ❓ ВОПРОС 1: КАК БУДЕТ ОБУЧАТЬСЯ БОТ?

### Онлайн обучение (как в текущем боте)

**Принцип:**
Бот обучается **на живых данных** после каждой закрытой позиции.

### Процесс обучения:

#### 1️⃣ **Запись результата после закрытия позиции**

Когда позиция закрывается (TP или SL):

```python
def on_position_closed(position: Position):
    """Вызывается когда позиция закрывается"""

    # Получаем outcome
    if position.exit_reason == 'TP':
        outcome = 1  # УСПЕХ ✅
    elif position.exit_reason == 'SL':
        outcome = 0  # НЕУДАЧА ❌
    else:  # Timeout
        outcome = None  # Пропускаем

    if outcome is None:
        return

    # Фазу рынка на момент входа
    phase = position.entry_context['phase']

    # Фичи на момент входа (сохранены при открытии)
    features = position.entry_features  # 68D

    # Передаем результат всем экспертам
    for expert in [xgb_expert, rf_expert, arf_expert, nn_expert]:
        expert.record_result(
            x_raw=features,
            y_up=outcome,  # 1 = TP (успех), 0 = SL (неудача)
            used_in_live=True,
            p_pred=position.p_up,  # наше предсказание при входе
            reg_ctx={'phase': phase}
        )

    # Передаем результат META
    meta.record_result(
        p_xgb=position.predictions['p_xgb'],
        p_rf=position.predictions['p_rf'],
        p_arf=position.predictions['p_arf'],
        p_nn=position.predictions['p_nn'],
        p_base=position.predictions['p_base'],
        y_up=outcome,
        phase=phase,
        used_in_live=True
    )
```

#### 2️⃣ **Переобучение экспертов**

**Триггер:** Каждые 100 закрытых позиций (или каждые 24 часа)

```python
def maybe_retrain_experts():
    """Переобучает экспертов если накопилось достаточно данных"""

    closed_count = get_closed_positions_since_last_training()

    if closed_count >= 100:  # Достаточно новых данных

        print("[Training] Переобучение экспертов...")

        # Переобучаем каждого эксперта по фазам
        for phase in range(6):

            # XGBoost: полное переобучение
            xgb_expert.maybe_train(ph=phase, reg_ctx={'phase': phase})

            # RandomForest: батч-дообучение
            rf_expert.maybe_train(ph=phase, reg_ctx={'phase': phase})

            # ARF: онлайн обучение (уже обновлен при record_result)
            # Ничего делать не нужно

            # NN: батч-дообучение с SGD
            nn_expert.maybe_train(ph=phase, reg_ctx={'phase': phase})

        # Сохраняем обновленные модели
        save_models()

        print("[Training] ✅ Эксперты переобучены!")
```

#### 3️⃣ **Переобучение META**

**Триггер:** Каждые 500 закрытых позиций (или раз в неделю)

```python
def maybe_retrain_meta():
    """Переобучает META если накопилось достаточно данных"""

    closed_count = meta.get_closed_count()

    if closed_count >= 500:  # Достаточно для CMA-ES

        print("[Training] Переобучение META через CMA-ES...")

        # Оптимизация весов нейросети через CMA-ES
        for phase in range(6):
            meta.train_phase(phase, n_iterations=50)

        # Сохраняем обновленную META
        save_meta()

        print("[Training] ✅ META переобучена!")
```

### Ключевые особенности обучения:

✅ **TP закрытия = SUCCESS (y=1)** для обучения
✅ **SL закрытия = FAILURE (y=0)** для обучения
✅ **Timeout = пропускаем** (не используем для обучения)

✅ **Онлайн обучение** - модели адаптируются к изменениям рынка
✅ **Фазовое обучение** - раздельные модели для каждой фазы рынка
✅ **Постепенное улучшение** - с каждой закрытой позицией модели становятся точнее

---

## ❓ ВОПРОС 2: LONG И SHORT ПОЗИЦИИ

### ⚠️ КРИТИЧНОЕ ДОПОЛНЕНИЕ!

В исходном плане я описал **только LONG** позиции. Но бот должен уметь открывать **и LONG и SHORT**!

### Логика направления сделки:

```python
def determine_direction(p_up: float, p_threshold: float) -> Optional[str]:
    """
    Определяет направление сделки на основе вероятности

    Args:
        p_up: Вероятность роста от META (0-1)
        p_threshold: Адаптивный порог (0.5 + δ)

    Returns:
        'LONG' - если p_up > p_threshold (ожидаем рост)
        'SHORT' - если p_up < (1 - p_threshold) (ожидаем падение)
        None - если неопределенность (пропускаем)
    """

    # Порог для SHORT (симметричный)
    p_threshold_short = 1.0 - p_threshold

    if p_up > p_threshold:
        # Сильная уверенность в росте → LONG
        return 'LONG'

    elif p_up < p_threshold_short:
        # Сильная уверенность в падении → SHORT
        return 'SHORT'

    else:
        # Неопределенность (0.48 < p_up < 0.52) → пропускаем
        return None


# Пример использования:
p_up = 0.62  # META предсказание
p_thr = 0.51 + delta  # Адаптивный порог (например, 0.54)

direction = determine_direction(p_up, p_thr)

if direction == 'LONG':
    # Открываем LONG
    # TP = entry + 1.2*ATR
    # SL = entry - 0.8*ATR
    open_long_position(...)

elif direction == 'SHORT':
    # Открываем SHORT
    # TP = entry - 1.2*ATR (цена падает → профит)
    # SL = entry + 0.8*ATR (цена растет → убыток)
    open_short_position(...)

else:
    # Пропускаем монету
    pass
```

### TP/SL для LONG и SHORT:

#### LONG позиция:
```python
entry_price = 100.0
atr = 5.0

tp_price = entry_price + 1.2 * atr  # 106.0 (рост на 6%)
sl_price = entry_price - 0.8 * atr  # 96.0 (падение на 4%)

# Прибыль при TP: +6%
# Убыток при SL: -4%
# R/R = 1.5
```

#### SHORT позиция:
```python
entry_price = 100.0
atr = 5.0

tp_price = entry_price - 1.2 * atr  # 94.0 (падение на 6% → профит)
sl_price = entry_price + 0.8 * atr  # 104.0 (рост на 4% → убыток)

# Прибыль при TP: +6%
# Убыток при SL: -4%
# R/R = 1.5 (тот же!)
```

### Target для обучения (с SHORT):

```python
def calculate_tp_sl_outcome_bidirectional(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    atr_value: float,
    direction: str,  # 'LONG' или 'SHORT'
    tp_mult: float = 1.2,
    sl_mult: float = 0.8,
    max_bars: int = 100
) -> Optional[dict]:
    """
    Определяет исход для LONG или SHORT позиции
    """

    if direction == 'LONG':
        tp_price = entry_price + tp_mult * atr_value
        sl_price = entry_price - sl_mult * atr_value

        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            if high >= tp_price:
                return {'outcome': 1, 'direction': 'LONG'}  # SUCCESS
            if low <= sl_price:
                return {'outcome': 0, 'direction': 'LONG'}  # FAILURE

    elif direction == 'SHORT':
        tp_price = entry_price - tp_mult * atr_value
        sl_price = entry_price + sl_mult * atr_value

        for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            if low <= tp_price:
                return {'outcome': 1, 'direction': 'SHORT'}  # SUCCESS
            if high >= sl_price:
                return {'outcome': 0, 'direction': 'SHORT'}  # FAILURE

    return None  # Timeout
```

### EV расчет для LONG/SHORT:

```python
def calculate_ev_bidirectional(p_up: float) -> tuple[float, float]:
    """
    Рассчитывает EV для LONG и SHORT

    Returns:
        (ev_long, ev_short)
    """

    # EV для LONG
    # P(TP) = p_up, Reward = 1.5R
    # P(SL) = 1 - p_up, Risk = 1.0R
    ev_long = p_up * 1.5 - (1 - p_up) * 1.0
    ev_long = p_up * 2.5 - 1.0

    # EV для SHORT
    # P(TP) = (1 - p_up), Reward = 1.5R
    # P(SL) = p_up, Risk = 1.0R
    ev_short = (1 - p_up) * 1.5 - p_up * 1.0
    ev_short = 1.5 - p_up * 2.5

    return ev_long, ev_short


# Пример:
p_up = 0.62  # META предсказание

ev_long, ev_short = calculate_ev_bidirectional(p_up)

print(f"EV LONG:  {ev_long:.4f}")   # +0.0500 (прибыльно)
print(f"EV SHORT: {ev_short:.4f}")  # -0.0500 (убыточно)

# Выбираем направление с максимальным EV
if ev_long > 0.02 and ev_long > ev_short:
    direction = 'LONG'
elif ev_short > 0.02 and ev_short > ev_long:
    direction = 'SHORT'
else:
    direction = None  # Пропускаем
```

### Отбор монет с учетом SHORT:

```python
def select_top_coins_bidirectional(
    all_pairs: List[str],
    experts: dict,
    meta: MetaNeuralCEM,
    top_n: int = 10
) -> List[dict]:
    """
    Отбирает топ-N возможностей (LONG или SHORT)
    """

    opportunities = []

    for symbol in all_pairs:
        # Фильтры ликвидности
        if not apply_filters(symbol):
            continue

        # Получаем предсказание META
        p_up = get_meta_prediction(symbol, experts, meta)

        # Рассчитываем EV для обоих направлений
        ev_long, ev_short = calculate_ev_bidirectional(p_up)

        # Выбираем лучшее направление
        if ev_long > 0.02 and ev_long > ev_short:
            opportunities.append({
                'symbol': symbol,
                'direction': 'LONG',
                'p_up': p_up,
                'ev': ev_long
            })

        elif ev_short > 0.02 and ev_short > ev_long:
            opportunities.append({
                'symbol': symbol,
                'direction': 'SHORT',
                'p_up': p_up,
                'ev': ev_short
            })

    # Сортируем по абсолютному EV
    opportunities.sort(key=lambda x: x['ev'], reverse=True)

    return opportunities[:top_n]


# Результат:
# [
#   {'symbol': 'BTCUSDT', 'direction': 'LONG', 'p_up': 0.67, 'ev': 0.0675},
#   {'symbol': 'ETHUSDT', 'direction': 'SHORT', 'p_up': 0.35, 'ev': 0.0625},
#   {'symbol': 'SOLUSDT', 'direction': 'LONG', 'p_up': 0.61, 'ev': 0.0525},
#   ...
# ]
```

---

## ❓ ВОПРОС 3: ПОРОГ ПРИНЯТИЯ РЕШЕНИЯ (p_thr + δ)

### Адаптация текущей логики

В текущем боте используется **адаптивный порог**: `p_thr + δ`

Где:
- **p_thr** - базовый порог (0.51 в начале)
- **δ (delta)** - динамическая добавка на основе производительности

### Перенос логики на Binance:

#### 1️⃣ **Базовый порог (p_thr)**

```python
def calculate_base_threshold(
    total_closed_trades: int,
    recent_hour_trades: int
) -> float:
    """
    Базовый порог как в текущем боте

    Логика из bnbusdrt6.py (строки 26-28):
    - Если < 500 сделок ИЛИ нет сделок за последний час → 0.51
    - Иначе → рассчитываем через EV с учетом комиссий
    """

    if total_closed_trades < 500 or recent_hour_trades == 0:
        return 0.51  # Консервативный порог в начале

    else:
        # Рассчитываем EV-оптимальный порог
        # Для Binance:
        # - Комиссия: 0.1% на вход + 0.1% на выход = 0.2%
        # - R/R = 1.5
        # - Break-even WR = 40%

        # Минимальная вероятность для прибыльности:
        # p_up * 1.5R - (1 - p_up) * 1.0R - 0.002 * capital > 0
        # p_up * 2.5 - 1.0 - 0.005 > 0  # 0.005 = комиссия в единицах R
        # p_up > 0.402

        # Добавляем небольшой запас:
        p_thr = 0.41  # Чуть выше break-even

        return p_thr
```

#### 2️⃣ **Адаптивная дельта (δ)**

Используем **текущую функцию** из bnbusdrt6.py:

```python
def adaptive_delta_protect(
    recent_wr: float,      # Винрейт за последние 100 сделок
    calib_error: float,    # Ошибка калибровки (ECE)
    n_trades: int          # Количество сделок за последний час
) -> float:
    """
    Динамическая δ на основе производительности

    ИЗ bnbusdrt6.py:348-383 (БЕЗ ИЗМЕНЕНИЙ!)
    """
    base_delta = 0.03

    # Масштаб по винрейту
    if recent_wr >= 0.58:
        wr_scale = 0.7      # Агрессивнее при отличном WR
    elif recent_wr >= 0.56:
        wr_scale = 0.85
    elif recent_wr >= 0.54:
        wr_scale = 1.0      # Базовый
    elif recent_wr >= 0.52:
        wr_scale = 1.2
    else:
        wr_scale = 1.4      # Осторожнее при плохом WR

    # Масштаб по калибровке
    if calib_error < 0.03:
        calib_scale = 0.9
    elif calib_error < 0.05:
        calib_scale = 1.0
    else:
        calib_scale = 1.15

    # Масштаб по активности
    activity_scale = 1.0 if n_trades >= 20 else 1.1

    delta_eff = base_delta * wr_scale * calib_scale * activity_scale
    return float(np.clip(delta_eff, 0.02, 0.08))
```

#### 3️⃣ **Финальный порог**

```python
def get_adaptive_threshold(
    total_closed: int,
    recent_hour: int,
    recent_wr: float,
    calib_error: float
) -> float:
    """
    Финальный адаптивный порог

    ТОЧНО КАК В ТЕКУЩЕМ БОТЕ!
    """

    # Базовый порог
    p_thr = calculate_base_threshold(total_closed, recent_hour)

    # Адаптивная дельта
    delta = adaptive_delta_protect(recent_wr, calib_error, recent_hour)

    # Финальный порог
    p_threshold_final = p_thr + delta

    # Ограничиваем разумными пределами
    p_threshold_final = np.clip(p_threshold_final, 0.45, 0.70)

    return p_threshold_final


# Пример использования:
total_closed = 1200       # Всего закрытых сделок
recent_hour = 15          # Сделок за последний час
recent_wr = 0.56          # Винрейт 56%
calib_error = 0.04        # ECE = 4%

p_threshold = get_adaptive_threshold(
    total_closed, recent_hour, recent_wr, calib_error
)

print(f"Адаптивный порог: {p_threshold:.4f}")
# Вывод: 0.4385 (0.41 + 0.0285)

# Теперь проверяем монету:
p_up = 0.62  # META предсказание

if p_up > p_threshold:
    print(f"✅ LONG: p_up={p_up:.3f} > threshold={p_threshold:.3f}")
elif p_up < (1 - p_threshold):
    print(f"✅ SHORT: p_up={p_up:.3f} < threshold={1-p_threshold:.3f}")
else:
    print(f"❌ SKIP: неопределенность")
```

### Визуализация порогов:

```
Распределение вероятностей:

0.0   0.2   0.4   0.5   0.6   0.8   1.0
|-----|-----|-----|-----|-----|-----|
      ◄───SHORT───►   ◄───LONG───►
                  ▲   ▲
                  │   │
        (1-p_thr) │   │ p_thr
                  │   │
         ┌────────┴───┴────────┐
         │   НЕОПРЕДЕЛЕННОСТЬ   │
         │      (SKIP ZONE)     │
         └──────────────────────┘

Пример при p_thr = 0.54:
- p_up > 0.54 → LONG
- p_up < 0.46 → SHORT  (1 - 0.54)
- 0.46 ≤ p_up ≤ 0.54 → SKIP
```

---

## 📊 ОБНОВЛЕННЫЙ WORKFLOW БОТА

### Полный цикл торговли:

```python
def trading_cycle():
    """Полный цикл торговли на Binance"""

    # ═══════════════════════════════════════════════════════
    # ШАГ 1: ОБНОВЛЕНИЕ АДАПТИВНОГО ПОРОГА
    # ═══════════════════════════════════════════════════════

    stats = get_trading_statistics()

    p_threshold = get_adaptive_threshold(
        total_closed=stats['total_closed'],
        recent_hour=stats['recent_hour_trades'],
        recent_wr=stats['win_rate_last_100'],
        calib_error=stats['calibration_error']
    )

    print(f"[Threshold] p_thr = {p_threshold:.4f}")

    # ═══════════════════════════════════════════════════════
    # ШАГ 2: СКРИНИНГ МОНЕТ (EV ФИЛЬТР)
    # ═══════════════════════════════════════════════════════

    all_pairs = get_all_usdt_pairs()  # ~400 монет

    print(f"[Screening] Анализ {len(all_pairs)} монет...")

    opportunities = select_top_coins_bidirectional(
        all_pairs=all_pairs,
        experts=experts,
        meta=meta,
        top_n=10
    )

    print(f"[Screening] Найдено {len(opportunities)} возможностей")

    # ═══════════════════════════════════════════════════════
    # ШАГ 3: ОЦЕНКА КАЖДОЙ ВОЗМОЖНОСТИ
    # ═══════════════════════════════════════════════════════

    for opp in opportunities:
        symbol = opp['symbol']
        direction = opp['direction']
        p_up = opp['p_up']
        ev = opp['ev']

        print(f"\n[{symbol}] Direction: {direction}")
        print(f"  p_up = {p_up:.4f}, EV = {ev:.4f}")

        # Проверяем порог
        if direction == 'LONG':
            if p_up <= p_threshold:
                print(f"  ❌ SKIP: p_up={p_up:.3f} ≤ threshold={p_threshold:.3f}")
                continue

        elif direction == 'SHORT':
            if p_up >= (1 - p_threshold):
                print(f"  ❌ SKIP: p_up={p_up:.3f} ≥ (1-threshold)={1-p_threshold:.3f}")
                continue

        # Прошел порог!
        print(f"  ✅ PASS threshold check")

        # ═══════════════════════════════════════════════════════
        # ШАГ 4: ОТКРЫТИЕ ПОЗИЦИИ
        # ═══════════════════════════════════════════════════════

        # Рассчитываем размер позиции (Kelly)
        position_size = calculate_position_size(
            capital=get_current_capital(),
            p_up=p_up,
            ev=ev,
            recent_wr=stats['win_rate_last_100']
        )

        # Получаем ATR
        atr = calculate_atr_4h(symbol)

        # Текущая цена
        entry_price = get_current_price(symbol)

        # Открываем позицию
        if direction == 'LONG':
            position = open_long_position(
                symbol=symbol,
                entry_price=entry_price,
                size=position_size,
                atr=atr,
                p_up=p_up,
                predictions={
                    'p_xgb': ...,
                    'p_rf': ...,
                    'p_arf': ...,
                    'p_nn': ...,
                    'p_base': ...
                }
            )

        elif direction == 'SHORT':
            position = open_short_position(
                symbol=symbol,
                entry_price=entry_price,
                size=position_size,
                atr=atr,
                p_up=p_up,
                predictions={...}
            )

        print(f"  🚀 Открыта {direction} позиция")
        print(f"     Entry: {entry_price:.2f}")
        print(f"     TP: {position.tp_price:.2f}")
        print(f"     SL: {position.sl_price:.2f}")
        print(f"     Size: ${position_size:.2f}")

    # ═══════════════════════════════════════════════════════
    # ШАГ 5: МОНИТОРИНГ ОТКРЫТЫХ ПОЗИЦИЙ
    # ═══════════════════════════════════════════════════════

    check_open_positions()

    # ═══════════════════════════════════════════════════════
    # ШАГ 6: ОБУЧЕНИЕ (если нужно)
    # ═══════════════════════════════════════════════════════

    maybe_retrain_experts()
    maybe_retrain_meta()
```

---

## 📋 ИТОГОВЫЕ ОТВЕТЫ

### ✅ ОТВЕТ 1: Обучение бота

**ДА, бот будет обучаться онлайн:**
- ✅ После каждой закрытой позиции записывается результат (TP=успех, SL=неудача)
- ✅ Переобучение экспертов: каждые 100 закрытых позиций
- ✅ Переобучение META: каждые 500 закрытых позиций
- ✅ Постоянная адаптация к изменениям рынка

### ✅ ОТВЕТ 2: LONG и SHORT позиции

**ДА, бот будет открывать LONG и SHORT:**
1. ✅ **Скрининг** → отбор топ-10 возможностей (с направлением)
2. ✅ **Эксперты + META** → дают вероятность p_up
3. ✅ **Логика направления:**
   - `p_up > p_threshold` → **LONG**
   - `p_up < (1 - p_threshold)` → **SHORT**
   - Иначе → SKIP (неопределенность)
4. ✅ Одинаковый R/R для LONG и SHORT (1.5:1)

### ✅ ОТВЕТ 3: Адаптивный порог (p_thr + δ)

**ДА, используется текущая логика:**
- ✅ **p_thr** = базовый порог (0.51 в начале, 0.41 после 500 сделок)
- ✅ **δ** = адаптивная дельта (функция `adaptive_delta_protect` БЕЗ ИЗМЕНЕНИЙ)
- ✅ **Финальный порог** = `p_thr + δ` (от 0.45 до 0.70)
- ✅ Адаптируется на основе:
  - Винрейта последних 100 сделок
  - Ошибки калибровки (ECE)
  - Активности торговли

---

## 🎯 ВАЖНО ДОБАВИТЬ В ПРОМТЫ

При реализации этапов нужно добавить в промты:

### Промт 3.2 (Target definition):
```
ВАЖНО: Добавить поддержку SHORT позиций!

Функция должна уметь рассчитывать outcome для LONG и SHORT:
- LONG: TP выше entry, SL ниже entry
- SHORT: TP ниже entry, SL выше entry

Датасет должен содержать примеры обоих направлений.
```

### Промт 4.1 (EV-фильтр):
```
ВАЖНО: EV-фильтр должен выбирать ЛУЧШЕЕ НАПРАВЛЕНИЕ (LONG или SHORT)!

Для каждой монеты рассчитывать:
- EV_long = p_up * 2.5 - 1.0
- EV_short = 1.5 - p_up * 2.5

Выбирать направление с максимальным положительным EV.
Результат: список из 10 возможностей с указанием направления.
```

### Промт 6.1 (Торговый бот):
```
ВАЖНО:
1. Использовать адаптивный порог p_thr + δ (как в текущем боте)
2. Поддержка LONG и SHORT позиций
3. Проверка порога ПЕРЕД открытием позиции

Логика:
- Если p_up > (p_thr + δ) → LONG
- Если p_up < (1 - p_thr - δ) → SHORT
- Иначе → SKIP
```

---

Нужно ли мне **обновить BINANCE_MIGRATION_PLAN.md** с этими дополнениями? Или создать отдельный файл `BINANCE_IMPLEMENTATION_DETAILS.md`?
