# ⚠️ КРИТИЧЕСКИ ВАЖНО: ВРЕМЕННАЯ СОГЛАСОВАННОСТЬ ДАННЫХ

**Версия:** 1.0
**Дата:** 2025-11-06
**Приоритет:** КРИТИЧЕСКИЙ

---

## 🚨 ПРОБЛЕМА: LOOK-AHEAD BIAS

### ❌ НЕПРАВИЛЬНАЯ СХЕМА (Temporal Leakage):

```python
# НЕПРАВИЛЬНО! ❌
def on_position_closed(position: Position):
    """Когда позиция закрывается"""

    # Получаем outcome
    outcome = 1 if position.exit_reason == 'TP' else 0

    # ❌ ОШИБКА: Рассчитываем фичи СЕЙЧАС (на момент закрытия)
    df_5m = get_ohlcv(position.symbol, '5m', 200)
    df_15m = get_ohlcv(position.symbol, '15m', 200)
    df_30m = get_ohlcv(position.symbol, '30m', 200)
    df_4h = get_ohlcv(position.symbol, '4h', 100)

    features = build_extended_features(df_5m, df_15m, df_30m, df_4h)

    # ❌ ПРОБЛЕМА: features содержат информацию о том, что цена УЖЕ достигла TP/SL!
    # Модель будет обучаться на информации из будущего

    # Обучаем экспертов
    xgb_expert.record_result(x_raw=features, y_up=outcome)
```

**Почему это плохо:**

1. **Фичи на момент закрытия** включают информацию о движении цены к TP/SL
2. Например, `momentum_5m` на момент закрытия может показывать сильный тренд вверх, если достигнут TP
3. Модель запоминает: "Если momentum высокий → TP достигнут" (очевидная корреляция)
4. **Но в реальной торговле** на момент входа momentum может быть совсем другим!
5. Модель не сможет генерализовать на новые данные

### ✅ ПРАВИЛЬНАЯ СХЕМА (Snapshot at Entry):

```python
# ПРАВИЛЬНО! ✅
def open_position(symbol: str, direction: str, ...):
    """Открываем позицию"""

    # 1. Получаем данные НА МОМЕНТ ПРИНЯТИЯ РЕШЕНИЯ
    df_5m = get_ohlcv(symbol, '5m', 200)
    df_15m = get_ohlcv(symbol, '15m', 200)
    df_30m = get_ohlcv(symbol, '30m', 200)
    df_4h = get_ohlcv(symbol, '4h', 100)

    # 2. Рассчитываем фичи НА МОМЕНТ ВХОДА
    features_at_entry = build_extended_features(df_5m, df_15m, df_30m, df_4h)

    # 3. Получаем предсказания экспертов НА МОМЕНТ ВХОДА
    p_xgb = xgb_expert.predict(features_at_entry)
    p_rf = rf_expert.predict(features_at_entry)
    p_arf = arf_expert.predict(features_at_entry)
    p_nn = nn_expert.predict(features_at_entry)
    p_base = base_model.predict(features_at_entry)

    # 4. Получаем финальное предсказание META НА МОМЕНТ ВХОДА
    ctx = build_regime_context(df_4h)
    p_meta = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base, ctx['phase'])

    # 5. СОХРАНЯЕМ СНИМОК ДАННЫХ в объекте Position
    position = Position(
        symbol=symbol,
        direction=direction,
        entry_time=datetime.now(),
        entry_price=current_price,

        # ✅ КРИТИЧЕСКИ ВАЖНО: Сохраняем снимок на момент входа!
        entry_snapshot={
            'features': features_at_entry.copy(),  # 68D вектор
            'predictions': {
                'p_xgb': p_xgb,
                'p_rf': p_rf,
                'p_arf': p_arf,
                'p_nn': p_nn,
                'p_base': p_base,
                'p_meta': p_meta
            },
            'context': {
                'phase': ctx['phase'],
                'trend_sign': ctx['trend_sign'],
                'vol_ratio': ctx['vol_ratio'],
                'atr': atr_value
            },
            'timestamp': datetime.now().timestamp(),
            'ohlcv_snapshot': {
                'last_close_5m': df_5m['close'].iloc[-1],
                'last_close_15m': df_15m['close'].iloc[-1],
                'last_close_30m': df_30m['close'].iloc[-1],
                'last_close_4h': df_4h['close'].iloc[-1]
            }
        },

        # TP/SL уровни
        tp_price=tp_price,
        sl_price=sl_price,
        ...
    )

    # 6. Открываем позицию на бирже
    exchange.create_order(...)

    return position


def on_position_closed(position: Position):
    """Когда позиция закрывается (TP или SL)"""

    # Получаем outcome
    outcome = 1 if position.exit_reason == 'TP' else 0

    # ✅ ПРАВИЛЬНО: Используем СОХРАНЕННЫЕ фичи на момент входа!
    features_at_entry = position.entry_snapshot['features']
    predictions = position.entry_snapshot['predictions']
    context = position.entry_snapshot['context']

    # Обучаем экспертов на данных МОМЕНТА ВХОДА
    xgb_expert.record_result(
        x_raw=features_at_entry,  # ✅ Фичи на момент входа!
        y_up=outcome,              # Результат после закрытия
        used_in_live=True,
        p_pred=predictions['p_xgb'],
        reg_ctx={'phase': context['phase']}
    )

    rf_expert.record_result(
        x_raw=features_at_entry,
        y_up=outcome,
        ...
    )

    # То же для всех экспертов и META
    meta.record_result(
        p_xgb=predictions['p_xgb'],
        p_rf=predictions['p_rf'],
        p_arf=predictions['p_arf'],
        p_nn=predictions['p_nn'],
        p_base=predictions['p_base'],
        y_up=outcome,
        phase=context['phase'],
        used_in_live=True
    )

    print(f"[Training] Recorded: {position.symbol} {position.direction}")
    print(f"  Entry time: {position.entry_time}")
    print(f"  Exit time: {position.exit_time}")
    print(f"  Duration: {position.duration_hours:.1f}h")
    print(f"  Outcome: {'TP ✅' if outcome == 1 else 'SL ❌'}")
```

---

## 📊 ВИЗУАЛИЗАЦИЯ ВРЕМЕННОЙ ЛИНИИ

```
═══════════════════════════════════════════════════════════════════════════

t0 (МОМЕНТ ВХОДА)              t1 (МОМЕНТ ВЫХОДА - через N часов/дней)
     ▼                                    ▼
     │                                    │
┌────┴────────────────────────────────────┴────┐
│    ПОЗИЦИЯ ОТКРЫТА                            │
│    BTC/USDT LONG                              │
│                                               │
│    Entry: $42,000                             │
│    TP: $44,632 (+6.27%)                       │
│    SL: $41,416 (-1.39%)                       │
└───────────────────────────────────────────────┘

В t0 (ВХОД):                    В t1 (ВЫХОД):
─────────────                   ─────────────
✅ Рассчитать фичи (68D)        ✅ Получить outcome (TP=1 или SL=0)
✅ Получить p_xgb, p_rf, ...    ✅ Использовать СОХРАНЕННЫЕ фичи из t0
✅ Получить p_meta              ✅ Обучить: X=фичи_t0, y=outcome_t1
✅ СОХРАНИТЬ snapshot           ✅ Никаких новых расчетов фич!
✅ Открыть позицию


═══════════════════════════════════════════════════════════════════════════
```

---

## 💾 СТРУКТУРА ДАННЫХ

### Position Dataclass (обновленная):

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import numpy as np

@dataclass
class Position:
    """Позиция с полным снимком данных на момент входа"""

    # Базовая информация
    symbol: str
    direction: str  # 'LONG' или 'SHORT'
    entry_time: datetime
    entry_price: float
    amount: float  # Количество монет
    position_value: float  # Стоимость в USDT

    # TP/SL уровни
    tp_price: float
    sl_price: float
    atr_value: float

    # Ордера
    entry_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None

    # ✅ КРИТИЧЕСКИЙ РАЗДЕЛ: Снимок на момент входа
    entry_snapshot: Dict = field(default_factory=dict)
    # Структура entry_snapshot:
    # {
    #   'features': np.ndarray (68D),
    #   'predictions': {
    #       'p_xgb': float,
    #       'p_rf': float,
    #       'p_arf': float,
    #       'p_nn': float,
    #       'p_base': float,
    #       'p_meta': float
    #   },
    #   'context': {
    #       'phase': int (0-5),
    #       'trend_sign': float,
    #       'trend_abs': float,
    #       'vol_ratio': float,
    #       'atr': float
    #   },
    #   'timestamp': float,
    #   'ohlcv_snapshot': {
    #       'last_close_5m': float,
    #       'last_close_15m': float,
    #       'last_close_30m': float,
    #       'last_close_4h': float
    #   }
    # }

    # Результат (заполняется при закрытии)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'TIMEOUT', 'MANUAL'
    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Статус
    status: str = 'OPEN'  # 'OPEN', 'CLOSED'

    @property
    def duration_hours(self) -> float:
        """Длительность удержания позиции в часах"""
        if self.exit_time is None:
            return (datetime.now() - self.entry_time).total_seconds() / 3600
        return (self.exit_time - self.entry_time).total_seconds() / 3600

    def to_dict(self) -> dict:
        """Конвертация в словарь для сохранения"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'amount': self.amount,
            'position_value': self.position_value,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price,
            'atr_value': self.atr_value,
            'entry_snapshot': {
                'features': self.entry_snapshot['features'].tolist(),  # numpy → list
                'predictions': self.entry_snapshot['predictions'],
                'context': self.entry_snapshot['context'],
                'timestamp': self.entry_snapshot['timestamp'],
                'ohlcv_snapshot': self.entry_snapshot['ohlcv_snapshot']
            },
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'status': self.status
        }
```

---

## 🔄 ПОЛНЫЙ WORKFLOW С ВРЕМЕННОЙ СОГЛАСОВАННОСТЬЮ

### 1️⃣ Открытие позиции (t0):

```python
def open_position_with_snapshot(
    symbol: str,
    direction: str,
    capital: float,
    experts: dict,
    meta: MetaNeuralCEM,
    base_model
) -> Position:
    """
    Открывает позицию с сохранением снимка данных

    ✅ ВСЕ расчеты делаются ОДИН РАЗ на момент входа
    ✅ Снимок сохраняется в Position для будущего обучения
    """

    print(f"\n[Entry] {symbol} {direction}")

    # ═══════════════════════════════════════════════════════
    # БЛОК 1: ПОЛУЧЕНИЕ ДАННЫХ (t0)
    # ═══════════════════════════════════════════════════════

    df_5m = get_ohlcv(symbol, '5m', 200)
    df_15m = get_ohlcv(symbol, '15m', 200)
    df_30m = get_ohlcv(symbol, '30m', 200)
    df_4h = get_ohlcv(symbol, '4h', 100)

    current_price = df_5m['close'].iloc[-1]

    # ═══════════════════════════════════════════════════════
    # БЛОК 2: РАСЧЕТ ФИЧЕЙ (t0)
    # ═══════════════════════════════════════════════════════

    features = build_extended_features(df_5m, df_15m, df_30m, df_4h)

    print(f"  Features calculated: {features.shape}")

    # ═══════════════════════════════════════════════════════
    # БЛОК 3: ПРЕДСКАЗАНИЯ ЭКСПЕРТОВ (t0)
    # ═══════════════════════════════════════════════════════

    ctx = build_regime_context(df_4h)
    phase = ctx['phase']

    p_xgb = experts['xgb'].proba_up(features, reg_ctx={'phase': phase})[0]
    p_rf = experts['rf'].proba_up(features, reg_ctx={'phase': phase})[0]
    p_arf = experts['arf'].proba_up(features, reg_ctx={'phase': phase})[0]
    p_nn = experts['nn'].proba_up(features, reg_ctx={'phase': phase})[0]

    p_base = base_model.predict(features)

    # ═══════════════════════════════════════════════════════
    # БЛОК 4: META ПРЕДСКАЗАНИЕ (t0)
    # ═══════════════════════════════════════════════════════

    p_meta = meta.predict(p_xgb, p_rf, p_arf, p_nn, p_base, phase)

    print(f"  Predictions: XGB={p_xgb:.3f}, RF={p_rf:.3f}, ARF={p_arf:.3f}, NN={p_nn:.3f}")
    print(f"  META: {p_meta:.3f}")

    # ═══════════════════════════════════════════════════════
    # БЛОК 5: РАСЧЕТ TP/SL (t0)
    # ═══════════════════════════════════════════════════════

    atr = calculate_atr(df_4h, period=10).iloc[-1]

    if direction == 'LONG':
        tp_price = current_price + 1.2 * atr
        sl_price = current_price - 0.8 * atr
    else:  # SHORT
        tp_price = current_price - 1.2 * atr
        sl_price = current_price + 0.8 * atr

    # ═══════════════════════════════════════════════════════
    # БЛОК 6: СОЗДАНИЕ СНИМКА (t0)
    # ═══════════════════════════════════════════════════════

    entry_snapshot = {
        'features': features.copy(),  # ✅ Копия массива
        'predictions': {
            'p_xgb': float(p_xgb),
            'p_rf': float(p_rf),
            'p_arf': float(p_arf),
            'p_nn': float(p_nn),
            'p_base': float(p_base),
            'p_meta': float(p_meta)
        },
        'context': {
            'phase': int(phase),
            'trend_sign': float(ctx['trend_sign']),
            'trend_abs': float(ctx['trend_abs']),
            'vol_ratio': float(ctx['vol_ratio']),
            'atr': float(atr)
        },
        'timestamp': time.time(),
        'ohlcv_snapshot': {
            'last_close_5m': float(df_5m['close'].iloc[-1]),
            'last_close_15m': float(df_15m['close'].iloc[-1]),
            'last_close_30m': float(df_30m['close'].iloc[-1]),
            'last_close_4h': float(df_4h['close'].iloc[-1])
        }
    }

    # ═══════════════════════════════════════════════════════
    # БЛОК 7: РАЗМЕР ПОЗИЦИИ (Kelly)
    # ═══════════════════════════════════════════════════════

    ev = p_meta * 2.5 - 1.0
    position_size_usd = calculate_kelly_size(capital, ev, p_meta)
    amount = position_size_usd / current_price

    # ═══════════════════════════════════════════════════════
    # БЛОК 8: СОЗДАНИЕ ОБЪЕКТА POSITION
    # ═══════════════════════════════════════════════════════

    position = Position(
        symbol=symbol,
        direction=direction,
        entry_time=datetime.now(),
        entry_price=current_price,
        amount=amount,
        position_value=position_size_usd,
        tp_price=tp_price,
        sl_price=sl_price,
        atr_value=atr,
        entry_snapshot=entry_snapshot,  # ✅ СНИМОК СОХРАНЕН!
        status='OPEN'
    )

    # ═══════════════════════════════════════════════════════
    # БЛОК 9: ОТКРЫТИЕ ПОЗИЦИИ НА БИРЖЕ
    # ═══════════════════════════════════════════════════════

    # Entry order
    entry_order = exchange.create_market_order(
        symbol=symbol,
        side='BUY' if direction == 'LONG' else 'SELL',
        amount=amount
    )
    position.entry_order_id = entry_order['id']

    # TP order
    tp_order = exchange.create_limit_order(
        symbol=symbol,
        side='SELL' if direction == 'LONG' else 'BUY',
        amount=amount,
        price=tp_price
    )
    position.tp_order_id = tp_order['id']

    # SL order
    sl_order = exchange.create_stop_loss_order(
        symbol=symbol,
        side='SELL' if direction == 'LONG' else 'BUY',
        amount=amount,
        price=sl_price
    )
    position.sl_order_id = sl_order['id']

    print(f"  ✅ Position opened:")
    print(f"     Entry: {current_price:.2f}")
    print(f"     TP: {tp_price:.2f} ({((tp_price/current_price-1)*100):.2f}%)")
    print(f"     SL: {sl_price:.2f} ({((sl_price/current_price-1)*100):.2f}%)")
    print(f"     Size: ${position_size_usd:.2f}")
    print(f"     Snapshot saved: {len(entry_snapshot['features'])} features")

    return position
```

### 2️⃣ Закрытие позиции и обучение (t1):

```python
def close_position_and_train(
    position: Position,
    exit_reason: str,  # 'TP', 'SL', 'TIMEOUT', 'MANUAL'
    exit_price: float,
    experts: dict,
    meta: MetaNeuralCEM
):
    """
    Закрывает позицию и обучает модели на данных момента входа

    ✅ Используются СОХРАНЕННЫЕ фичи из entry_snapshot
    ✅ НЕТ расчетов новых фичей на момент выхода
    """

    print(f"\n[Exit] {position.symbol} {position.direction} - {exit_reason}")

    # ═══════════════════════════════════════════════════════
    # БЛОК 1: ОБНОВЛЕНИЕ ПОЗИЦИИ
    # ═══════════════════════════════════════════════════════

    position.exit_time = datetime.now()
    position.exit_price = exit_price
    position.exit_reason = exit_reason
    position.status = 'CLOSED'

    # Расчет PnL
    if position.direction == 'LONG':
        position.pnl = (exit_price - position.entry_price) * position.amount
        position.pnl_pct = (exit_price / position.entry_price - 1) * 100
    else:  # SHORT
        position.pnl = (position.entry_price - exit_price) * position.amount
        position.pnl_pct = (1 - exit_price / position.entry_price) * 100

    print(f"  Duration: {position.duration_hours:.1f}h")
    print(f"  PnL: ${position.pnl:.2f} ({position.pnl_pct:+.2f}%)")

    # ═══════════════════════════════════════════════════════
    # БЛОК 2: ОПРЕДЕЛЕНИЕ OUTCOME
    # ═══════════════════════════════════════════════════════

    if exit_reason == 'TP':
        outcome = 1  # УСПЕХ ✅
    elif exit_reason == 'SL':
        outcome = 0  # НЕУДАЧА ❌
    else:  # TIMEOUT или MANUAL
        outcome = None  # Пропускаем

    if outcome is None:
        print(f"  ⚠️  Skipping training (exit reason: {exit_reason})")
        return

    print(f"  Outcome: {'SUCCESS ✅' if outcome == 1 else 'FAILURE ❌'}")

    # ═══════════════════════════════════════════════════════
    # БЛОК 3: ИЗВЛЕЧЕНИЕ СНИМКА НА МОМЕНТ ВХОДА
    # ═══════════════════════════════════════════════════════

    snapshot = position.entry_snapshot

    features_at_entry = snapshot['features']  # ✅ Фичи момента входа!
    predictions = snapshot['predictions']
    context = snapshot['context']

    print(f"  Using snapshot from entry:")
    print(f"    Timestamp: {datetime.fromtimestamp(snapshot['timestamp'])}")
    print(f"    Phase: {context['phase']}")
    print(f"    Features shape: {features_at_entry.shape}")

    # ═══════════════════════════════════════════════════════
    # БЛОК 4: ОБУЧЕНИЕ ЭКСПЕРТОВ
    # ═══════════════════════════════════════════════════════

    phase = context['phase']

    # XGBoost
    experts['xgb'].record_result(
        x_raw=features_at_entry,       # ✅ Снимок t0
        y_up=outcome,                  # Результат t1
        used_in_live=True,
        p_pred=predictions['p_xgb'],
        reg_ctx={'phase': phase}
    )

    # RandomForest
    experts['rf'].record_result(
        x_raw=features_at_entry,
        y_up=outcome,
        used_in_live=True,
        p_pred=predictions['p_rf'],
        reg_ctx={'phase': phase}
    )

    # ARF
    experts['arf'].record_result(
        x_raw=features_at_entry,
        y_up=outcome,
        used_in_live=True,
        p_pred=predictions['p_arf'],
        reg_ctx={'phase': phase}
    )

    # NN
    experts['nn'].record_result(
        x_raw=features_at_entry,
        y_up=outcome,
        used_in_live=True,
        p_pred=predictions['p_nn'],
        reg_ctx={'phase': phase}
    )

    # ═══════════════════════════════════════════════════════
    # БЛОК 5: ОБУЧЕНИЕ META
    # ═══════════════════════════════════════════════════════

    meta.record_result(
        p_xgb=predictions['p_xgb'],
        p_rf=predictions['p_rf'],
        p_arf=predictions['p_arf'],
        p_nn=predictions['p_nn'],
        p_base=predictions['p_base'],
        y_up=outcome,
        phase=phase,
        used_in_live=True
    )

    print(f"  ✅ Training data recorded for all models")

    # ═══════════════════════════════════════════════════════
    # БЛОК 6: СОХРАНЕНИЕ В CSV ДЛЯ АНАЛИЗА
    # ═══════════════════════════════════════════════════════

    save_trade_to_csv({
        'symbol': position.symbol,
        'direction': position.direction,
        'entry_time': position.entry_time.isoformat(),
        'exit_time': position.exit_time.isoformat(),
        'entry_price': position.entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': position.pnl,
        'pnl_pct': position.pnl_pct,
        'duration_hours': position.duration_hours,
        'outcome': outcome,
        'p_meta': predictions['p_meta'],
        'phase': phase
    })
```

---

## ✅ ПРОВЕРКА ПРАВИЛЬНОСТИ

### Тест 1: Размер фичей

```python
# При открытии
position = open_position_with_snapshot(...)
print(f"Features shape: {position.entry_snapshot['features'].shape}")
# Output: (68,)  ✅

# При закрытии
close_position_and_train(position, 'TP', exit_price, ...)
# Внутри функции:
features = position.entry_snapshot['features']
print(f"Features shape: {features.shape}")
# Output: (68,)  ✅ ТЕ ЖЕ фичи!
```

### Тест 2: Временные метки

```python
# При открытии
position = open_position_with_snapshot(...)
t0 = position.entry_snapshot['timestamp']
print(f"Entry snapshot timestamp: {datetime.fromtimestamp(t0)}")
# Output: 2025-11-06 14:30:00

# Через 2 дня при закрытии
close_position_and_train(position, 'TP', ...)
t1 = time.time()
print(f"Exit timestamp: {datetime.fromtimestamp(t1)}")
# Output: 2025-11-08 16:45:00

print(f"Time difference: {(t1 - t0) / 3600:.1f} hours")
# Output: 50.25 hours

# ✅ Используем фичи из t0, outcome из t1
```

### Тест 3: Значения фичей не меняются

```python
# При открытии
position = open_position_with_snapshot(...)
features_original = position.entry_snapshot['features'].copy()
print(f"Feature[0] at entry: {features_original[0]:.6f}")

# Сохраняем в файл
save_position(position)

# Через 2 дня загружаем и закрываем
position = load_position(position_id)
features_loaded = position.entry_snapshot['features']
print(f"Feature[0] at exit: {features_loaded[0]:.6f}")

# Проверка
assert np.allclose(features_original, features_loaded), "Features changed!"
# ✅ PASS
```

---

## 📝 ЧЕКЛИСТ РЕАЛИЗАЦИИ

При реализации убедись что:

- [ ] Position dataclass содержит поле `entry_snapshot: Dict`
- [ ] При открытии позиции вызывается `build_extended_features()` ОДИН РАЗ
- [ ] Все предсказания экспертов делаются на момент входа
- [ ] Снимок сохраняется в Position перед открытием ордера
- [ ] Position сохраняется в файл (JSON) для персистентности
- [ ] При закрытии НЕ вызывается `build_extended_features()`
- [ ] Обучение использует `position.entry_snapshot['features']`
- [ ] Outcome определяется только по exit_reason (TP/SL)
- [ ] В CSV/логах сохраняется timestamp снимка
- [ ] Тесты проверяют что фичи не меняются между t0 и t1

---

## 🚨 АНТИПАТТЕРНЫ (НЕ ДЕЛАТЬ!)

### ❌ Антипаттерн 1: Расчет фичей при закрытии

```python
# НЕТ! ❌
def on_position_closed(position):
    df = get_ohlcv(position.symbol, '5m', 200)
    features = build_extended_features(df)  # ❌ ПЛОХО!
    expert.record_result(features, outcome)
```

### ❌ Антипаттерн 2: Использование последних цен

```python
# НЕТ! ❌
def on_position_closed(position):
    current_close = get_current_price(position.symbol)  # ❌ ПЛОХО!
    features['close'] = current_close
    expert.record_result(features, outcome)
```

### ❌ Антипаттерн 3: Смешивание t0 и t1 данных

```python
# НЕТ! ❌
def on_position_closed(position):
    # Берем часть фичей из snapshot
    features_old = position.entry_snapshot['features'][:50]

    # Дорасчитываем остальные СЕЙЧАС
    df = get_ohlcv(...)
    features_new = calculate_new_features(df)  # ❌ ПЛОХО!

    features = np.concatenate([features_old, features_new])
    expert.record_result(features, outcome)
```

---

## 📖 SUMMARY

### ✅ ПРАВИЛЬНЫЙ ПОДХОД:

1. **t0 (ВХОД):**
   - Получить OHLCV данные
   - Рассчитать фичи (68D)
   - Получить предсказания
   - **СОХРАНИТЬ снимок в Position**
   - Открыть позицию

2. **t1 (ВЫХОД):**
   - Позиция закрылась (TP/SL)
   - Outcome = 1 (TP) или 0 (SL)
   - **ИЗВЛЕЧЬ снимок из Position**
   - Обучить: X=фичи_t0, y=outcome_t1

### 🎯 КЛЮЧЕВЫЕ ПРИНЦИПЫ:

- ✅ **Temporal consistency:** фичи и outcome из РАЗНЫХ моментов времени
- ✅ **No look-ahead bias:** модель видит только прошлое на момент входа
- ✅ **Snapshot pattern:** "заморозка" состояния рынка
- ✅ **Immutable features:** фичи не меняются после сохранения
- ✅ **Reproducible:** можно воспроизвести решение по snapshot

---

**КРИТИЧЕСКИ ВАЖНО ДЛЯ ПРАВИЛЬНОГО ОБУЧЕНИЯ!** ⚠️
