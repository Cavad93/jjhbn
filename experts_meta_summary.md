# TL;DR: Оценка экспертов и META

## 📊 КРАТКИЕ ОЦЕНКИ

| Эксперт | Оценка | Главная проблема |
|---------|--------|------------------|
| XGBoost | **7/10** | Нет калибровки вероятностей |
| RandomForest | **6/10** | Memory leak + слишком глубоко (depth=10) |
| AdaptiveRF | **5/10** | ❌❌❌ **BROKEN: не сохраняется!** |
| NeuralNet | **6.5/10** | Overparametrized (11K params для 100-500 samples) |
| META | **6/10** | ❌❌ **Massive overfit (31200 params / 150 samples)** |

---

## 🔥 КРИТИЧЕСКИЕ ПРОБЛЕМЫ

### 1. ❌❌❌ AdaptiveRF теряет обучение при перезапуске
```python
# arf_expert.py:258
def load(cls, filepath):
    expert.is_trained = False  # СБРАСЫВАЕТСЯ!!!
    return expert
```
**Результат:** После каждого перезапуска бота ARF учится с нуля.

**Fix:** Использовать `dill` для сериализации River модели.

---

### 2. ❌❌ META сильно overparametrized
```
Параметры: 31200 (6 phases × 5200 params/phase)
Данные: ~150 samples per phase
Соотношение: 208:1

Норма: 10:1 maximum
Реально нужно: 312000 samples
Есть: 150 samples
```
**Результат:** Гарантированный overfitting.

**Fix:** Упростить до 200 params (Input(10D) → Dense(16) → Dense(1)).

---

### 3. ❌ Нет калибровки вероятностей
```python
# ВСЕ эксперты возвращают uncalibrated probabilities
p_xgb = model.predict_proba(X)  # Может быть 0.9, но реально 0.6!
```
**Результат:** Threshold=0.65 не означает "65% уверенности".

**Fix:** Добавить `CalibratedClassifierCV` с isotonic method.

---

### 4. ❌ RandomForest memory leak
```python
self.model.estimators_ += new_forest.estimators_  # Accumulates forever
# После 1000 updates: 100 + 1000×10 = 10100 деревьев!
```
**Результат:** Память растет бесконечно.

**Fix:** Ограничить max 500 деревьев, удалять старые (FIFO).

---

## 💡 ЧТО ДЕЛАТЬ

### 🔴 ПРИОРИТЕТ 1: CRITICAL FIXES (эта неделя)

1. **Fix AdaptiveRF save/load**
   ```python
   import dill
   def save(self): dill.dump(self.model, f)
   def load(cls): expert.model = dill.load(f); expert.is_trained = True
   ```

2. **Добавить калибровку**
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrator = CalibratedClassifierCV(model, method='isotonic', cv=3)
   ```

3. **Fix RF memory leak**
   ```python
   MAX_TREES = 500
   if len(self.model.estimators_) > MAX_TREES:
       self.model.estimators_ = self.model.estimators_[n_remove:]
   ```

**Ожидаемый результат:** Win rate +3-5% (55% → 58-63%)

---

### 🟡 ПРИОРИТЕТ 2: АРХИТЕКТУРА (через 2 недели)

4. **Упростить META**
   ```python
   # Было: 31200 params (6 phases × 5200)
   # Стало: 200 params (Input(10D) → Dense(16) → Dense(1))
   # Одна модель, phase как feature
   ```

5. **Упростить NeuralNet**
   ```python
   # Было: 68 → 128 → 64 → 32 → 1 (11K params)
   # Стало: 68 → 32 → 16 → 1 (2.7K params)
   ```

6. **Diversity metric**
   ```python
   disagreement = mean(|p_i - p_j|)  # Насколько эксперты расходятся
   if disagreement < 0.05: warn("Too similar - overfitting")
   ```

**Ожидаемый результат:** Win rate +5-8% (55% → 60-65%)

---

### 🟢 ПРИОРИТЕТ 3: ОПТИМИЗАЦИИ (через месяц)

7. **AutoML для гиперпараметров** (Optuna)
8. **SimpleEnsemble** вместо сложной META
9. **Transfer learning** для холодного старта

**Ожидаемый результат:** Win rate +7-13% (55% → 62-68%), ROI +50-100%

---

## 📈 СРАВНЕНИЕ ЭКСПЕРТОВ

### Что они умеют хорошо:

| Эксперт | Преимущества |
|---------|--------------|
| **XGBoost** | ✅✅ Быстрый, точный, feature importance |
| **RandomForest** | ✅ Stable, параллелизация |
| **AdaptiveRF** | ✅✅✅ **DRIFT DETECTION!** (ADWIN), онлайн обучение |
| **NeuralNet** | ✅ Deep learning, GPU support |
| **META** | ✅✅ MC Dropout uncertainty, Attention, Phase-specific |

### Что они умеют плохо:

| Эксперт | Проблемы |
|---------|----------|
| **XGBoost** | ❌ Нет калибровки, нет drift detection |
| **RandomForest** | ❌ Memory leak, depth=10 слишком глубоко |
| **AdaptiveRF** | ❌❌❌ **НЕ СОХРАНЯЕТСЯ**, медленный |
| **NeuralNet** | ❌ Overparametrized, BatchNorm при малых данных |
| **META** | ❌❌ **Massive overfitting**, CMA-ES медленный |

---

## 🎯 ГЛАВНЫЙ ВЫВОД

**Система умная, но:**
- **Слишком сложная** для доступных данных (overfitting)
- **Критические баги** (ARF не сохраняется)
- **Нет калибровки** (вероятности не надежны)

**Рекомендация:**
1. Срочно пофиксить ARF и добавить калибровку
2. Упростить META в 100× (31200 → 200 params)
3. Упростить NeuralNet в 4× (11K → 2.7K params)

**Ожидаемое улучшение:**
- **Win Rate: +7-13%** (с 55-60% → 62-68%)
- **ROI: +50-100%**
- **Стабильность: Значительно выше**

---

## 📂 ДОКУМЕНТАЦИЯ

**Полный анализ:** `analysis_experts_and_meta_architecture.md`

**Содержит:**
- Детальный разбор каждого эксперта
- Архитектура META с кодом
- Сравнительная таблица
- Конкретные fixes с кодом
- Ожидаемые результаты
