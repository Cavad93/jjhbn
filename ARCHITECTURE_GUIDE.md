# Архитектура системы принятия решений бота

## 📐 Общая схема системы

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ВХОДНЫЕ ДАННЫЕ                                      │
│  Market Data: Price, Volume, OrderBook, Funding, Basis, Indicators (68D)   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  FEATURE EXTRACTION           │
                    │  68-мерный вектор фич         │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        │                           │                           │
┌───────▼──────┐            ┌──────▼──────┐            ┌───────▼────────┐
│  BASE LOGIC  │            │  ML EXPERTS │            │  CONTEXT       │
│              │            │             │            │                │
│ • M (Mean    │            │ 1. XGBoost  │            │ • Phase (0-5)  │
│   Reversion) │            │ 2. RF       │            │ • Volatility   │
│ • S (Short   │            │ 3. ARF      │            │ • Trend        │
│   Term)      │            │ 4. NN       │            │ • Jump flag    │
│ • B (Bid/Ask)│            │             │            │ • Funding      │
│ • R (RSI)    │            │             │            │ • Book imb     │
│              │            │             │            │ • OFI, Basis   │
└───────┬──────┘            └──────┬──────┘            └───────┬────────┘
        │                           │                           │
        │   p_base                  │ p_xgb, p_rf,             │ context
        │   (0.0-1.0)              │ p_arf, p_nn               │ (7D)
        │                           │ (4 вероятности)           │
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                            ┌───────▼──────────┐
                            │   META LAYER     │
                            │                  │
                            │ SimplifiedMETA   │
                            │ (24 params) ИЛИ  │
                            │                  │
                            │ NeuralMETA       │
                            │ (31,200 params)  │
                            │                  │
                            │ • Phase-specific │
                            │ • SHADOW/ACTIVE  │
                            │ • Drift detect   │
                            └───────┬──────────┘
                                    │
                            p_final (0.0-1.0)
                                    │
                            ┌───────▼──────────┐
                            │  DECISION LOGIC  │
                            │                  │
                            │ if p_final > 0.5:│
                            │    LONG          │
                            │ else:            │
                            │    SHORT/SKIP    │
                            └───────┬──────────┘
                                    │
                            ┌───────▼──────────┐
                            │  POSITION MGMT   │
                            │                  │
                            │ • Size calc      │
                            │ • Risk mgmt      │
                            │ • TP/SL          │
                            └──────────────────┘
```

---

## 🎯 Детальное описание компонентов

### 1. BASE LOGIC (Базовая логика)

**Что это**: Набор простых правил без ML

**Компоненты**:
- **M (Mean Reversion)**: Ловит откат к средней
- **S (Short Term)**: Краткосрочный моментум
- **B (Bid/Ask)**: Дисбаланс заявок
- **R (RSI)**: Индикатор перекупленности/перепроданности

**Как работает**:
```python
# Псевдокод
def base_logic(features):
    signals = {
        'M': mean_reversion_signal(features),  # -1 to 1
        'S': short_term_signal(features),      # -1 to 1
        'B': bid_ask_signal(features),         # -1 to 1
        'R': rsi_signal(features)              # -1 to 1
    }

    # Взвешенная комбинация
    p_base = weighted_average(signals)

    # Нормализация в [0, 1]
    p_base = sigmoid(p_base)

    return p_base  # Например: 0.52
```

**Выход**: `p_base` - вероятность роста по базовой логике (0.0 - 1.0)

**Пример**:
```
Рынок: BTC падает, но RSI oversold (30)
M signal: +0.8 (ожидается откат вверх)
S signal: -0.4 (краткосрочный downtrend)
B signal: +0.3 (больше bid orders)
R signal: +0.9 (oversold → reversal)

p_base = sigmoid(0.8*0.3 + (-0.4)*0.2 + 0.3*0.2 + 0.9*0.3) = 0.63
```

---

### 2. ML EXPERTS (4 эксперта)

**Что это**: Машинное обучение для предсказания вероятности достижения TP

#### 2.1 XGBoost Expert

**Тип**: Gradient Boosting Decision Trees

**Архитектура**:
- 100 деревьев
- Max depth: 5
- Learning rate: 0.1
- L1/L2 регуляризация

**Как работает**:
```python
# Обучение
xgb.fit(X_historical, y_historical)  # y=1 если TP достигнут, иначе 0

# После fix: калибровка
calibrator.fit(X_val, y_val)

# Предсказание
p_xgb_raw = xgb.predict_proba(X_new)  # Может быть 0.82
p_xgb = calibrator.transform(p_xgb_raw)  # Калиброванная: 0.67
```

**Обучение**:
- **Initial**: Полное обучение на накопленных данных
- **Partial fit**: Добавление 10 новых деревьев каждые 100 примеров
- **Калибровка**: Обновляется при каждом полном обучении

**Пример**:
```
Входные фичи (68D):
- price_change_5m: -0.02
- volume_ratio: 1.5
- rsi_14: 35
- macd: -0.001
- ... (64+ других)

XGBoost анализ:
Tree 1: "если rsi < 40 → +0.3"
Tree 2: "если volume_ratio > 1.2 → +0.2"
...
Tree 100: "если price_change < -0.01 → +0.1"

Raw prediction: 0.82
После калибровки: 0.67  ← Более надежная оценка
```

#### 2.2 RandomForest Expert

**Тип**: Ensemble of Decision Trees с Bagging

**Архитектура**:
- 100+ деревьев (растет онлайн)
- Max depth: 6 (было 10, но это overfitting)
- Max total trees: 500 (fix memory leak)
- Bootstrap sampling

**Как работает**:
```python
# Обучение
rf.fit(X_historical, y_historical)

# Partial fit (добавление новых деревьев)
rf.partial_fit(X_new, y_new, n_new_trees=10)

# FIFO удаление если > 500 деревьев
if len(rf.estimators_) > 500:
    rf.estimators_ = rf.estimators_[10:]  # Удаляем старые

# Предсказание с калибровкой
p_rf = rf.predict_proba(X_new)  # Калиброванная вероятность
```

**Обучение**:
- **Initial**: 100 деревьев
- **Partial fit**: +10 деревьев каждые 100 примеров
- **Memory fix**: FIFO удаление при > 500 деревьев
- **Рекалибровка**: Каждые 5 partial_fit

**Пример**:
```
100 деревьев голосуют:
Tree 1: 0.7 (UP)
Tree 2: 0.4 (DOWN)
Tree 3: 0.8 (UP)
...
Tree 100: 0.6 (UP)

Raw average: 0.65
После калибровки: 0.61
```

#### 2.3 Adaptive RF (ARF) Expert

**Тип**: Online Random Forest с drift detection

**Архитектура**:
- 10 Hoeffding Trees (River library)
- ADWIN drift detector
- Истинное онлайн обучение (sample-by-sample)

**Как работает**:
```python
# Истинное онлайн обучение
for x, y in stream:
    arf.learn_one(x, y)  # Обучение на каждом примере

    # ADWIN детектит drift и пересоздает деревья при необходимости

# После FIX: сохранение с dill
dill.dump(arf.model, file)  # Сохраняет River модель
arf.is_trained = True  # НЕ сбрасывается!

# Предсказание
p_arf = arf.predict_proba(X_new)
```

**Обучение**:
- **Incremental**: Каждый пример обновляет модель
- **ADWIN**: Автоматически обнаруживает drift и адаптируется
- **Пересоздание деревьев**: При обнаружении drift

**Пример**:
```
Рынок меняется (drift):
Sample 1-1000: Модель обучена на bull market
Sample 1001: Drift detected! (переход в bear)
→ ADWIN пересоздает старые деревья
→ Модель адаптируется к новому режиму

p_arf = 0.58  (уже адаптировалась)
```

#### 2.4 Neural Network Expert

**Тип**: Deep Learning (PyTorch)

**Архитектура СТАРАЯ**:
- 68 → 128 → 64 → 32 → 1
- ~11,000 параметров
- BatchNorm + Dropout

**Архитектура НОВАЯ (SimplifiedNN)**:
- 68 → 32 → 16 → 1
- ~2,800 параметров
- LayerNorm + Dropout
- ReduceLROnPlateau scheduler

**Как работает**:
```python
# Обучение
nn.fit(X_train, y_train, X_val, y_val)
# Early stopping при застое validation loss

# Partial fit (fine-tuning)
nn.partial_fit(X_new, y_new, n_epochs=5)

# Предсказание
p_nn = nn.predict_proba(X_new)  # Sigmoid на выходе
```

**Обучение**:
- **Initial**: Полное обучение 20-50 эпох с early stopping
- **Partial fit**: Fine-tuning 3-5 эпох на новых данных
- **LR scheduling**: Автоматическое уменьшение при застое

**Пример**:
```
Forward pass:
Input (68D): [0.02, -0.01, 1.5, ...]
↓ Layer 1: 68 → 32
↓ ReLU + Dropout
↓ Layer 2: 32 → 16
↓ ReLU + Dropout
↓ Output: 16 → 1
↓ Sigmoid
Output: 0.64

NN видит сложные нелинейные паттерны, которые деревья могут пропустить
```

---

### 3. META LAYER (Мета-уровень)

**Что это**: Комбинирует предсказания экспертов в финальное решение

#### 3.1 SimplifiedEnsembleMETA (НОВАЯ, рекомендуется)

**Архитектура**:
```python
# Phase-specific веса
ensembles = {
    0: PhaseEnsemble([w_xgb=0.35, w_rf=0.25, w_arf=0.20, w_nn=0.20]),
    1: PhaseEnsemble([w_xgb=0.30, w_rf=0.30, w_arf=0.15, w_nn=0.25]),
    2: PhaseEnsemble([w_xgb=0.40, w_rf=0.20, w_arf=0.15, w_nn=0.25]),
    3: PhaseEnsemble([w_xgb=0.25, w_rf=0.35, w_arf=0.20, w_nn=0.20]),
    4: PhaseEnsemble([w_xgb=0.30, w_rf=0.25, w_arf=0.25, w_nn=0.20]),
    5: PhaseEnsemble([w_xgb=0.35, w_rf=0.25, w_arf=0.15, w_nn=0.25])
}
# Всего: 6 фаз × 4 веса = 24 параметра
```

**Как работает**:
```python
# 1. Определяем фазу рынка
phase = phase_from_ctx(context)  # 0-5

# 2. Получаем веса для этой фазы
ensemble = ensembles[phase]
weights = ensemble.weights  # [0.35, 0.25, 0.20, 0.20]

# 3. Взвешенное усреднение
p_final = (
    weights[0] * p_xgb +   # 0.35 * 0.67
    weights[1] * p_rf +    # 0.25 * 0.61
    weights[2] * p_arf +   # 0.20 * 0.58
    weights[3] * p_nn      # 0.20 * 0.64
)
# p_final = 0.6355

# 4. Нормализация
p_final = p_final / sum(weights)  # Веса нормализованы к 1.0
```

**Обучение**:
```python
# Каждые 100 примеров для фазы
def optimize_weights(phase):
    # Objective: minimize log loss
    def loss(weights):
        predictions = weights @ expert_predictions  # Матричное умножение
        return log_loss(y_true, predictions)

    # Scipy optimize (<1 секунда!)
    result = scipy.optimize.minimize(loss, x0=[0.25]*4, bounds=[(0,1)]*4)

    ensembles[phase].weights = result.x
```

**SHADOW/ACTIVE режимы**:
```python
if mode == "SHADOW":
    # Обучается, но не используется для решений
    # Накапливает статистику win rate
    if shadow_wr >= 0.58 and n_samples >= 80:
        mode = "ACTIVE"  # Переключение!

if mode == "ACTIVE":
    # Используется для предсказаний
    if active_wr < 0.52 or drift_detected:
        mode = "SHADOW"  # Возврат в безопасный режим
```

**Пример работы**:
```
Фаза 2 (sideways market):
Веса: [0.40, 0.20, 0.15, 0.25]

Предсказания экспертов:
p_xgb = 0.67  (калиброванная)
p_rf = 0.61   (калиброванная)
p_arf = 0.58
p_nn = 0.64

Взвешенное:
p_final = 0.40*0.67 + 0.20*0.61 + 0.15*0.58 + 0.25*0.64
        = 0.268 + 0.122 + 0.087 + 0.160
        = 0.637

Режим: ACTIVE (WR=62% на последних 100)
Решение: LONG (p_final > 0.5)
```

#### 3.2 NeuralMETA (СТАРАЯ, overfitting)

**Архитектура**:
```
Per-phase network (5200 params):
  Feature Engineering: 18D → 36D
  Attention Module: 7D → 16D → 4D (196 params)
  Main MLP: 36D → 64D → 32D → 16D → 1D (4993 params)
  MC Dropout: 30 passes

Total: 6 phases × 5200 = 31,200 parameters
```

**Проблемы**:
- ❌ 31,200 params / 150 samples = 208:1 ratio (overfitting!)
- ❌ Обучение 10-30 минут (CMA-ES)
- ❌ MC Dropout 30× медленнее
- ❌ Непрозрачная (черный ящик)

---

## 🔄 Полный цикл принятия решения

### Пример 1: Успешная сделка

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ВРЕМЯ: 14:30:00 UTC
РЫНОК: BTC/USDT
ЦЕНА: $42,150
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 1: Сбор данных
────────────────────
Market Data:
- Price: $42,150
- Change 5m: -0.3% (падение)
- Volume: 1.5× среднего (повышенный)
- RSI(14): 32 (oversold!)
- MACD: -0.002 (negative)
- Book imbalance: 0.6 (больше bid orders)
- Funding: -0.01% (short positions платят)

Context:
- Phase: 2 (sideways market)
- Volatility: medium
- Trend: slightly bearish

Features (68D): [-0.003, 1.5, 0.32, -0.002, 0.6, ...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 2: BASE LOGIC
──────────────────
M (Mean Reversion): +0.8  ← RSI oversold, ожидается откат
S (Short Term): -0.3      ← Краткосрочный downtrend
B (Bid/Ask): +0.4         ← Больше bid orders
R (RSI): +0.9             ← Сильный oversold сигнал

p_base = sigmoid(0.8*0.3 - 0.3*0.2 + 0.4*0.2 + 0.9*0.3)
       = sigmoid(0.53)
       = 0.63

Вывод BASE: Умеренный LONG сигнал

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 3: ML EXPERTS
──────────────────

XGBoost:
  Raw prediction: 0.78
  После калибровки: 0.71  ← Видит паттерн "oversold → reversal"
  Confidence: High

RandomForest:
  Raw prediction: 0.72
  После калибровки: 0.68  ← Подтверждает паттерн
  Confidence: Medium

AdaptiveRF:
  Prediction: 0.65  ← Недавно адаптировался к рынку
  Confidence: Medium

NeuralNet:
  Prediction: 0.70  ← Видит нелинейные зависимости
  Confidence: High

Статистика:
- Все эксперты > 0.5 (согласие на LONG)
- Std dev: 0.025 (низкий разброс → хорошо!)
- Diversity: OK (disagreement = 0.06)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 4: META LAYER (SimplifiedEnsemble)
────────────────────────────────────────
Phase: 2 (sideways)
Режим: ACTIVE (WR=64% на последних 100)

Веса для фазы 2:
- w_xgb: 0.40  ← XGBoost самый надежный в sideways
- w_rf:  0.20
- w_arf: 0.15
- w_nn:  0.25

Расчет:
p_final = 0.40*0.71 + 0.20*0.68 + 0.15*0.65 + 0.25*0.70
        = 0.284 + 0.136 + 0.0975 + 0.175
        = 0.6925

Вывод META: Сильный LONG сигнал (69.25%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 5: РЕШЕНИЕ
───────────────
Threshold: 0.5
p_final: 0.6925 > 0.5 ✅

РЕШЕНИЕ: OPEN LONG

Position Size:
- Risk: 1% от депозита
- Leverage: 3x
- Entry: $42,150
- TP: $42,570 (+1%)
- SL: $41,940 (-0.5%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

РЕЗУЛЬТАТ (через 15 минут):
───────────────────────────
ЦЕНА: $42,580 ✅
TP ДОСТИГНУТ!
PnL: +1% * 3x = +3%
Outcome: 1 (успех)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 6: ОБУЧЕНИЕ
────────────────
Новый пример добавлен в датасет:
X: [features из этой сделки]
y: 1 (TP достигнут)

Обновление экспертов:
✅ XGBoost: добавлен в buffer (partial_fit через 100 примеров)
✅ RandomForest: добавлен в buffer
✅ ARF: learn_one(X, y=1) ← немедленное обучение!
✅ NN: добавлен в buffer
✅ Calibrators: обновлены

Обновление META:
✅ Добавлен в историю фазы 2
✅ Hit rate обновлен: 64% → 64.5%
✅ Режим остается ACTIVE

Обновление DiversityMonitor:
✅ Disagreement: 0.06 (OK)
✅ Alert: None
```

### Пример 2: Неудачная сделка (обучение на ошибках)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ВРЕМЯ: 18:45:00 UTC
РЫНОК: BTC/USDT
ЦЕНА: $43,200
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 1-4: (аналогично)

Предсказания экспертов:
p_xgb = 0.62
p_rf = 0.58
p_arf = 0.48  ← ARF не согласен!
p_nn = 0.64

Diversity warning:
⚠️ HIGH_DISAGREEMENT (ARF vs others)
Std dev: 0.065 (высокий разброс)

META (Phase 3, ACTIVE):
p_final = 0.30*0.62 + 0.35*0.58 + 0.20*0.48 + 0.15*0.64
        = 0.186 + 0.203 + 0.096 + 0.096
        = 0.581

РЕШЕНИЕ: OPEN LONG (0.581 > 0.5, но близко к порогу)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

РЕЗУЛЬТАТ:
──────────
ЦЕНА: $43,030 ❌
SL ДОСТИГНУТ!
PnL: -0.5% * 3x = -1.5%
Outcome: 0 (неудача)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ЭТАП 6: ОБУЧЕНИЕ НА ОШИБКЕ
──────────────────────────
Новый пример:
X: [features]
y: 0 (SL достигнут)

Обновление экспертов:
✅ XGBoost: учится на ошибке (y=0)
✅ RandomForest: учится на ошибке
✅ ARF: learn_one(X, y=0) ← был прав, что предсказал 0.48!
✅ NN: учится на ошибке
✅ Calibrators: обновлены (будут более консервативны)

Обновление META:
✅ Добавлен в историю фазы 3
✅ Hit rate обновлен: 58% → 57.5%
⚠️ Hit rate близок к exit_wr (52%)

ADWIN drift detection:
✅ Проверка: miss добавлен в статистику
⚠️ Если серия промахов → drift detection → ACTIVE → SHADOW

Diversity analysis:
✅ ARF был прав - его вес может увеличиться при следующей оптимизации
✅ XGB/RF/NN ошиблись - их веса могут уменьшиться
```

---

## 🎓 Процесс обучения системы

### Жизненный цикл

```
┌─────────────────────────────────────────────────────────────────┐
│                    ЖИЗНЕННЫЙ ЦИКЛ СИСТЕМЫ                       │
└─────────────────────────────────────────────────────────────────┘

1. ХОЛОДНЫЙ СТАРТ (0-100 примеров)
   ├─ BASE LOGIC: работает сразу ✅
   ├─ ML Experts: возвращают 0.5 (не обучены) ❌
   ├─ META: использует простое среднее экспертов ⚠️
   └─ Решение: в основном BASE LOGIC

   Решение: Transfer Learning! ✨
   └─ Pretrain на исторических данных
      └─ Эксперты работают с первого дня ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. НАЧАЛЬНОЕ ОБУЧЕНИЕ (100-500 примеров)
   ├─ XGBoost: первое полное обучение (100 примеров) ✅
   ├─ RandomForest: первое полное обучение ✅
   ├─ ARF: обучается с каждого примера ✅
   ├─ NN: первое обучение (200 примеров) ✅
   ├─ Calibrators: обучение калибраторов ✅
   └─ META: начинает работать (150+ примеров) ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. ОНЛАЙН ОБУЧЕНИЕ (500+ примеров)

   Каждые 100 примеров:
   ├─ XGBoost: partial_fit (+10 деревьев)
   ├─ RandomForest: partial_fit (+10 деревьев, FIFO если >500)
   ├─ ARF: непрерывное обучение на каждом примере
   ├─ NN: partial_fit (3-5 эпох fine-tuning)
   └─ Calibrators: рекалибровка

   Каждые 100 примеров для фазы:
   └─ META: оптимизация весов (<1 секунда)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. АДАПТАЦИЯ К DRIFT (когда рынок меняется)

   ARF ADWIN:
   └─ Обнаружил drift → пересоздал деревья ✅

   META SHADOW/ACTIVE:
   ├─ ACTIVE режим: WR падает 64% → 55% → 51%
   ├─ WR < exit_wr (52%) или ADWIN drift
   └─ Переключение: ACTIVE → SHADOW ⚠️

   В SHADOW режиме:
   ├─ META обучается, но не используется
   ├─ Решения принимаются простым average
   ├─ Накапливается новая статистика
   └─ Когда WR >= 58% → SHADOW → ACTIVE ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. СТАБИЛЬНАЯ РАБОТА (1000+ примеров)

   Все компоненты работают оптимально:
   ├─ Эксперты обучены и калиброваны ✅
   ├─ META оптимизирована для всех фаз ✅
   ├─ ACTIVE режим с хорошим WR ✅
   ├─ Diversity monitor отслеживает ✅
   └─ Адаптация к drift автоматическая ✅
```

---

## 📊 Сравнение двух META подходов

```
┌─────────────────────────────────────────────────────────────────────────┐
│               SimplifiedEnsemble vs NeuralMETA                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SimplifiedEnsemble           │  NeuralMETA (старая)                  │
│  ════════════════════          │  ═══════════════════                  │
│                                 │                                       │
│  Параметры: 24                 │  Параметры: 31,200                    │
│  (6 фаз × 4 веса)              │  (6 фаз × 5,200)                      │
│                                 │                                       │
│  Обучение: <1 секунда          │  Обучение: 10-30 минут                │
│  (scipy.optimize)              │  (CMA-ES)                             │
│                                 │                                       │
│  Overfitting: Нет              │  Overfitting: Да ❌                    │
│  (ratio 24:150 = 1:6)          │  (ratio 31200:150 = 208:1)            │
│                                 │                                       │
│  Inference: Мгновенный         │  Inference: Медленный                 │
│  (простое умножение)           │  (MC Dropout 30×)                     │
│                                 │                                       │
│  Интерпретируемость: ✅        │  Интерпретируемость: ❌               │
│  (видны веса)                  │  (черный ящик)                        │
│                                 │                                       │
│  Формула:                      │  Формула:                             │
│  p = Σ(w_i × p_i)             │  p = NN(features, attention, MC)      │
│                                 │                                       │
│  Веса фазы 2:                  │  Веса: ???                            │
│  XGB: 0.40                     │  (невозможно интерпретировать)        │
│  RF:  0.20                     │                                       │
│  ARF: 0.15                     │                                       │
│  NN:  0.25                     │                                       │
│                                 │                                       │
│  Win Rate: 60-65%              │  Win Rate: 55-60%                     │
│  (меньше overfitting)          │  (overfitting снижает)                │
│                                 │                                       │
└─────────────────────────────────────────────────────────────────────────┘

РЕКОМЕНДАЦИЯ: SimplifiedEnsemble для production
```

---

## 🔧 Ключевые параметры системы

```python
# BASE LOGIC
base_weights = {
    'M': 0.3,  # Mean reversion
    'S': 0.2,  # Short term
    'B': 0.2,  # Bid/ask
    'R': 0.3   # RSI
}

# ML EXPERTS
xgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'partial_fit_trees': 10,  # Каждые 100 примеров
    'retrain_every': 100
}

rf_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'max_total_trees': 500,  # Memory leak fix
    'partial_fit_trees': 10,
    'retrain_every': 100
}

arf_params = {
    'n_models': 10,
    'grace_period': 100,
    # Online learning - обучается на каждом примере
}

nn_params = {
    'architecture': [68, 32, 16, 1],  # SimplifiedNN
    'batch_size': 'adaptive',  # 8-64
    'learning_rate': 0.001,
    'early_stopping': 10,
    'retrain_every': 100,
    'partial_fit_epochs': 5
}

# CALIBRATION
calibrator_params = {
    'method': 'auto',  # isotonic или sigmoid
    'min_samples_isotonic': 100,
    'val_split': 0.2  # 20% для калибровки
}

# META (SimplifiedEnsemble)
meta_params = {
    'n_phases': 6,
    'min_ready': 80,      # Минимум примеров для работы
    'min_train': 150,     # Минимум для обучения
    'retrain_every': 100, # Переобучение каждые 100
    'enter_wr': 0.58,     # SHADOW → ACTIVE
    'exit_wr': 0.52,      # ACTIVE → SHADOW
    'adwin_delta': 0.002  # Чувствительность drift
}

# DIVERSITY MONITOR
diversity_params = {
    'window_size': 100,
    'low_diversity_threshold': 0.05,
    'high_disagreement_threshold': 0.20,
    'high_correlation_threshold': 0.95
}
```

---

## 💡 Ключевые инсайты

1. **Калибровка критична**: Без нее threshold=0.65 не означает 65% вероятность
2. **ARF уникален**: Единственный с drift detection и истинным онлайн обучением
3. **META должна быть простой**: 24 параметра работают лучше чем 31,200
4. **SHADOW/ACTIVE защищает**: Автоматическое отключение при деградации
5. **Diversity важна**: Если эксперты слишком похожи → overfitting
6. **Transfer learning решает холодный старт**: Pretrain → работа с первого дня
7. **Phase-specific веса**: Разные фазы рынка требуют разных стратегий

---

Это полная архитектура системы! 🎯
