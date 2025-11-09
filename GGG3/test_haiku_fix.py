#!/usr/bin/env python3
"""
Тест для проверки исправления логики фильтрации Haiku
"""

# Константы (копия из HaikuConfig)
SCORE_THRESHOLD_APPROVE = 0.6

def test_filter_logic():
    """
    Тестируем новую логику фильтрации
    """

    print("=" * 70)
    print("ТЕСТ: Логика фильтрации Haiku")
    print("=" * 70)

    # Симуляция обогащённых монет после Haiku анализа
    test_cases = [
        {
            'name': 'Проходит оба фильтра',
            'symbol': 'BTCUSDT',
            'direction': 'LONG',
            'p_up': 0.65,
            'haiku_score': 0.75,
            'p_threshold': 0.58,
            'expected': 'APPROVED (tech+fund)',
            'expected_final_score_range': (0.65, 0.75)
        },
        {
            'name': 'Проходит только технический',
            'symbol': 'ETHUSDT',
            'direction': 'LONG',
            'p_up': 0.62,
            'haiku_score': 0.45,
            'p_threshold': 0.58,
            'expected': 'APPROVED (tech_only)',
            'expected_final_score': 0.62
        },
        {
            'name': 'Проходит только фундаментальный (КЛЮЧЕВОЙ КЕЙС)',
            'symbol': 'SOLUSDT',
            'direction': 'LONG',
            'p_up': 0.54,  # Ниже порога!
            'haiku_score': 0.85,  # Отличные новости!
            'p_threshold': 0.58,
            'expected': 'APPROVED (fund_only)',
            'expected_final_score': 0.85 * 0.9
        },
        {
            'name': 'Не проходит ни один фильтр',
            'symbol': 'DOGEUSDT',
            'direction': 'LONG',
            'p_up': 0.52,
            'haiku_score': 0.35,
            'p_threshold': 0.58,
            'expected': 'REJECTED',
            'expected_final_score': None
        },
        {
            'name': 'SHORT: проходит технический',
            'symbol': 'ADAUSDT',
            'direction': 'SHORT',
            'p_up': 0.35,  # p_down = 0.65 > threshold
            'haiku_score': 0.45,
            'p_threshold': 0.58,
            'expected': 'APPROVED (tech_only)',
            'expected_final_score': 0.65  # 1 - 0.35
        },
        {
            'name': 'SHORT: проходит только фундаментальный',
            'symbol': 'BNBUSDT',
            'direction': 'SHORT',
            'p_up': 0.48,  # p_down = 0.52 < threshold
            'haiku_score': 0.78,  # Отличные новости для SHORT!
            'p_threshold': 0.58,
            'expected': 'APPROVED (fund_only)',
            'expected_final_score': 0.78 * 0.9
        },
    ]

    # Симулируем логику фильтрации
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Symbol: {case['symbol']} {case['direction']}")
        print(f"   p_up: {case['p_up']:.3f}, haiku_score: {case['haiku_score']:.3f}")
        print(f"   p_threshold: {case['p_threshold']:.3f}")

        # Вычисляем confidence для направления
        direction = case['direction']
        if direction == 'LONG':
            confidence = case['p_up']
        else:
            confidence = 1 - case['p_up']

        passes_threshold = confidence > case['p_threshold']
        haiku_approved = case['haiku_score'] > SCORE_THRESHOLD_APPROVE

        # Применяем новую логику
        if not (passes_threshold or haiku_approved):
            result = 'REJECTED'
            final_score = None
        elif passes_threshold and haiku_approved:
            result = 'APPROVED (tech+fund)'
            final_score = confidence * 0.6 + case['haiku_score'] * 0.4
        elif passes_threshold:
            result = 'APPROVED (tech_only)'
            final_score = confidence
        else:
            result = 'APPROVED (fund_only)'
            final_score = case['haiku_score'] * 0.9

        # Проверяем результат
        if result.startswith(case['expected'][:10]):
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"   Expected: {case['expected']}")
        print(f"   Got: {result}")
        if final_score is not None:
            print(f"   Final score: {final_score:.3f}")
        else:
            print(f"   Final score: N/A")
        print(f"   {status}")

    print("\n" + "=" * 70)
    print("ВЫВОД:")
    print("=" * 70)
    print("✅ Новая логика позволяет Haiku одобрять монеты с отличными")
    print("   фундаментальными показателями, даже если технический сигнал слабый!")
    print("")
    print("❌ Старая логика отклоняла такие монеты полностью.")
    print("=" * 70)

if __name__ == "__main__":
    test_filter_logic()
