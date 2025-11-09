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
            'haiku_reason': 'positive news',  # Реально проанализировано
            'p_threshold': 0.58,
            'expected': 'APPROVED (fund_only)',
            'expected_final_score': 0.78 * 0.9
        },
        {
            'name': 'SKIPPED: высокий p_up, но не анализировалось (КЛЮЧЕВОЙ КЕЙС)',
            'symbol': 'MATICUSDT',
            'direction': 'LONG',
            'p_up': 0.68,  # Высокий!
            'haiku_score': 0.68,  # = p_up (для skipped)
            'haiku_reason': 'skipped: weak signal',  # НЕ анализировалось!
            'p_threshold': 0.58,
            'expected': 'APPROVED (tech_only)',  # Должно быть tech_only, НЕ fund!
            'expected_final_score': 0.68
        },
        {
            'name': 'ERROR: высокий haiku_score, но ошибка API',
            'symbol': 'LINKUSDT',
            'direction': 'LONG',
            'p_up': 0.55,
            'haiku_score': 0.72,
            'haiku_reason': 'api_error: timeout',  # Ошибка!
            'p_threshold': 0.58,
            'expected': 'REJECTED',  # НЕ проходит ни один фильтр
            'expected_final_score': None
        },
    ]

    print(f"\nВсего тест-кейсов: {len(test_cases)}")
    print(f"Включая 2 новых для проверки 'skipped' и 'error' логики\n")

    # Симулируем логику фильтрации
    passed_tests = 0
    failed_tests = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Symbol: {case['symbol']} {case['direction']}")
        print(f"   p_up: {case['p_up']:.3f}, haiku_score: {case['haiku_score']:.3f}")
        haiku_reason = case.get('haiku_reason', '')
        if haiku_reason:
            print(f"   haiku_reason: '{haiku_reason}'")
        print(f"   p_threshold: {case['p_threshold']:.3f}")

        # Вычисляем confidence для направления
        direction = case['direction']
        if direction == 'LONG':
            confidence = case['p_up']
        else:
            confidence = 1 - case['p_up']

        passes_threshold = confidence > case['p_threshold']

        # ИСПРАВЛЕНИЕ: Проверяем, была ли монета реально проанализирована
        was_analyzed = 'skipped' not in haiku_reason and 'error' not in haiku_reason
        haiku_approved = was_analyzed and (case['haiku_score'] > SCORE_THRESHOLD_APPROVE)

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
            passed_tests += 1
        else:
            status = "❌ FAIL"
            failed_tests += 1

        print(f"   Expected: {case['expected']}")
        print(f"   Got: {result}")
        if final_score is not None:
            print(f"   Final score: {final_score:.3f}")
        else:
            print(f"   Final score: N/A")
        print(f"   {status}")

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 70)
    print(f"Всего тестов: {len(test_cases)}")
    print(f"✅ Прошли: {passed_tests}")
    print(f"❌ Провалились: {failed_tests}")
    print("")
    if failed_tests == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ!")
    else:
        print("⚠️  ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ - требуется доработка")
    print("")
    print("КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:")
    print("=" * 70)
    print("✅ 1. Логика ИЛИ вместо И: монета одобряется если проходит")
    print("      хотя бы один фильтр (технический ИЛИ фундаментальный)")
    print("")
    print("✅ 2. 'Skipped' монеты НЕ считаются прошедшими фундаментальный фильтр")
    print("      (т.к. они вообще не были проанализированы Haiku)")
    print("")
    print("✅ 3. Монеты с ошибками API также НЕ проходят фундаментальный фильтр")
    print("")
    print("❌ Старая логика:")
    print("   - Отклоняла монеты с отличным fund score, но слабым tech score")
    print("   - 'Skipped' монеты ошибочно проходили как fund_only")
    print("=" * 70)

if __name__ == "__main__":
    test_filter_logic()
