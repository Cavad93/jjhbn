#!/usr/bin/env python3
"""
Генератор реалистичных синтетических данных для PancakeSwap Prediction

ОСНОВАН НА РЕАЛЬНЫХ ПАТТЕРНАХ:
- BNB цена: $500-$700
- 6 фаз рынка (по волатильности)
- Реалистичные тренды, флэты, развороты
- UP/DOWN соотношение ~52/48 (как в реальности)
- Фичи: momentum, volatility, RSI, volume
"""
import json
import numpy as np
from datetime import datetime

class RealisticDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

        # Параметры рынка
        self.base_price = 600.0  # BNB ~$600
        self.current_price = self.base_price

        # Фазы рынка (по волатильности)
        self.phases = {
            0: {'name': 'ultra_low', 'vol': 0.003, 'trend_strength': 0.2},
            1: {'name': 'low', 'vol': 0.007, 'trend_strength': 0.3},
            2: {'name': 'medium', 'vol': 0.012, 'trend_strength': 0.4},
            3: {'name': 'medium_high', 'vol': 0.018, 'trend_strength': 0.5},
            4: {'name': 'high', 'vol': 0.025, 'trend_strength': 0.6},
            5: {'name': 'extreme', 'vol': 0.040, 'trend_strength': 0.7}
        }

        self.current_phase = 1
        self.trend = 0.0  # -1 to 1
        self.trend_duration = 0

        # История цен
        self.price_history = [self.base_price] * 100

    def _update_trend(self):
        """Обновление тренда (меняется каждые 10-50 раундов)"""
        self.trend_duration += 1

        if self.trend_duration > np.random.randint(10, 50):
            # Новый тренд
            self.trend = np.random.uniform(-1, 1)
            self.trend_duration = 0

            # Смена фазы рынка (иногда)
            if np.random.rand() < 0.2:
                self.current_phase = np.random.choice([0, 1, 2, 3, 4, 5],
                                                       p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])

    def _generate_price_movement(self):
        """Генерация реалистичного движения цены"""
        phase_params = self.phases[self.current_phase]

        # Тренд компонента
        trend_move = self.trend * phase_params['trend_strength'] * 0.001

        # Шум (волатильность)
        noise = np.random.randn() * phase_params['vol']

        # Mean reversion к базовой цене
        mean_reversion = (self.base_price - self.current_price) * 0.001

        # Итоговое изменение
        price_change = trend_move + noise + mean_reversion

        return price_change

    def _calculate_features(self):
        """Расчет фич на основе истории цен"""
        closes = np.array(self.price_history[-50:])

        # Momentum
        momentum_5m = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        momentum_10m = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        momentum_20m = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0

        # Volatility
        volatility_10 = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) >= 10 else 0
        volatility_20 = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0

        # RSI
        deltas = np.diff(closes[-14:])
        gains = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        rs = gains / losses if losses > 0 else 1
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        # ATR
        high_low = np.random.rand() * volatility_10 * closes[-1]
        atr = high_low

        return {
            'momentum_5m': float(momentum_5m),
            'momentum_10m': float(momentum_10m),
            'momentum_20m': float(momentum_20m),
            'volatility_10': float(volatility_10),
            'volatility_20': float(volatility_20),
            'rsi': float(rsi),
            'bb_position': float(bb_position),
            'atr': float(atr),
            'price': float(self.current_price)
        }

    def generate_round(self, epoch):
        """Генерация одного раунда"""
        # Обновляем тренд
        self._update_trend()

        # Стартовая цена
        lock_price = self.current_price

        # Движение цены за 5 минут
        price_change = self._generate_price_movement()
        self.current_price = max(50, lock_price * (1 + price_change))  # Мин $50

        close_price = self.current_price

        # Обновляем историю
        self.price_history.append(close_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)

        # Outcome
        outcome = 1 if close_price > lock_price else 0

        # Расчет фич
        features = self._calculate_features()

        # Bull/Bear amounts (симуляция)
        total_amount = np.random.uniform(50, 500)  # BNB
        if outcome == 1:
            # UP раунд - больше ставок на bull
            bull_ratio = np.random.uniform(0.45, 0.70)
        else:
            # DOWN раунд - больше ставок на bear
            bull_ratio = np.random.uniform(0.30, 0.55)

        bull_amount = total_amount * bull_ratio
        bear_amount = total_amount * (1 - bull_ratio)

        return {
            'epoch': epoch,
            'timestamp': int(datetime.now().timestamp()) + epoch * 300,  # +5 min
            'lock_price': float(lock_price),
            'close_price': float(close_price),
            'outcome': int(outcome),
            'phase': int(self.current_phase),
            'features': features,
            'bull_amount': float(bull_amount),
            'bear_amount': float(bear_amount),
            'trend': float(self.trend)
        }

    def generate_dataset(self, n_rounds=5000):
        """Генерация полного датасета"""
        print(f"🎲 Генерация {n_rounds:,} реалистичных раундов...")

        rounds = []
        for epoch in range(1, n_rounds + 1):
            round_data = self.generate_round(epoch)
            rounds.append(round_data)

            if epoch % 500 == 0:
                print(f"  Сгенерировано: {epoch:,}/{n_rounds:,}", end='\r')

        print(f"\n✅ Сгенерировано {len(rounds):,} раундов")

        # Статистика
        outcomes = [r['outcome'] for r in rounds]
        up_pct = sum(outcomes) / len(outcomes) * 100

        phases = [r['phase'] for r in rounds]
        phase_dist = {p: sum(1 for ph in phases if ph == p) for p in range(6)}

        print(f"\n📊 СТАТИСТИКА:")
        print(f"  UP раундов: {sum(outcomes):,} ({up_pct:.1f}%)")
        print(f"  DOWN раундов: {len(outcomes) - sum(outcomes):,} ({100-up_pct:.1f}%)")
        print(f"\n  Распределение по фазам:")
        for phase, count in phase_dist.items():
            print(f"    Фаза {phase}: {count:,} ({count/len(rounds)*100:.1f}%)")

        return rounds

def main():
    print("="*60)
    print("ГЕНЕРАЦИЯ РЕАЛИСТИЧНЫХ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("="*60)

    generator = RealisticDataGenerator(seed=42)

    # Генерируем 60000 раундов (достаточно для обучения META Neural с 5200 параметров)
    rounds = generator.generate_dataset(n_rounds=60000)

    # Сохраняем
    output_file = 'realistic_synthetic_60k.json'
    print(f"\n💾 Сохранение в {output_file}...")

    data = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'total_rounds': len(rounds),
            'generator': 'RealisticDataGenerator v1.0',
            'seed': 42
        },
        'rounds': rounds
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Данные сохранены!")
    print(f"\n📁 Файл: {output_file}")
    print(f"📊 Размер: {len(json.dumps(data)) / 1024 / 1024:.1f} MB")

    return rounds

if __name__ == "__main__":
    import time
    start = time.time()
    rounds = main()
    elapsed = time.time() - start

    print(f"\n⏱️  Время: {elapsed:.1f}s")
    print(f"\n💡 Следующий шаг: Обучение экспертов и META Neural на этих данных")
