#!/usr/bin/env python3
"""
Примеры реализации различных стратегий управления позициями при ребалансировке
"""

from datetime import datetime, timedelta


# ============================================================================
# СТРАТЕГИЯ 1: "Полная свобода" (NO CLOSURE)
# ============================================================================

def strategy_1_no_closure(position_manager, top_20_opportunities):
    """
    Самая простая стратегия - не закрывать позиции принудительно.
    Позиции закроются только по TP или SL.

    Ожидаемое улучшение: +3.16%
    """
    print(f"\n  [Strategy 1: NO CLOSURE]")
    print(f"  All positions will close on TP/SL only")
    print(f"  Open positions: {len(position_manager.get_all_open())}")

    # НЕ закрываем ничего!
    closed_count = 0

    return closed_count


# ============================================================================
# СТРАТЕГИЯ 2: "Умное закрытие" (SMART CLOSURE)
# ============================================================================

def strategy_2_smart_closure(position_manager, top_20_opportunities, exchange, haiku_analyzer=None):
    """
    Закрывать только если есть объективные причины

    Ожидаемое улучшение: +2.5% до +3.5%
    """
    print(f"\n  [Strategy 2: SMART CLOSURE]")

    top_20_symbols = {opp['symbol'] for opp in top_20_opportunities}
    top_50_symbols = {opp['symbol'] for opp in top_20_opportunities[:50]} if len(top_20_opportunities) >= 50 else top_20_symbols

    closed_count = 0

    for position in list(position_manager.get_all_open()):
        if position.symbol in top_20_symbols:
            continue  # В топе - не трогаем

        # Получаем текущую цену и PnL
        current_price = exchange.get_current_price(position.symbol)
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        # Возраст позиции
        position_age = datetime.now() - position.entry_time
        age_hours = position_age.total_seconds() / 3600

        # ПРИЧИНЫ ДЛЯ ЗАКРЫТИЯ:

        # 1. Большой убыток (близко к типичному SL -7%)
        if pnl_pct < -5.0:
            print(f"  ❌ Closing {position.symbol}: Heavy loss {pnl_pct:.2f}%")
            position_manager.close_position(position.symbol, reason='MANUAL_HEAVY_LOSS')
            closed_count += 1
            continue

        # 2. Старая убыточная позиция
        if age_hours > 48 and pnl_pct < 0:
            print(f"  ⏰ Closing {position.symbol}: Old loss {age_hours:.1f}h, {pnl_pct:.2f}%")
            position_manager.close_position(position.symbol, reason='MANUAL_OLD_LOSS')
            closed_count += 1
            continue

        # 3. Упала далеко из топа
        if position.symbol not in top_50_symbols and pnl_pct < -2.0:
            print(f"  📉 Closing {position.symbol}: Out of TOP-50 + loss {pnl_pct:.2f}%")
            position_manager.close_position(position.symbol, reason='MANUAL_OUT_TOP50')
            closed_count += 1
            continue

        # 4. RED FLAG от Haiku (если доступен)
        if haiku_analyzer:
            analysis = haiku_analyzer.get_cached_analysis(position.symbol)
            if analysis and analysis.get('fundamental_score', 0.5) < 0.3:
                print(f"  🚩 Closing {position.symbol}: Haiku RED FLAG")
                position_manager.close_position(position.symbol, reason='MANUAL_RED_FLAG')
                closed_count += 1
                continue

        # Если ни одно условие не сработало - оставляем позицию
        print(f"  ✅ Keeping {position.symbol}: No critical issues (age={age_hours:.1f}h, pnl={pnl_pct:.2f}%)")

    return closed_count


# ============================================================================
# СТРАТЕГИЯ 3: "Временная толерантность" (TIME-BASED)
# ============================================================================

def strategy_3_time_based(position_manager, top_20_opportunities, exchange):
    """
    Дать позициям время "созреть" перед закрытием

    Ожидаемое улучшение: +2.8% до +3.3%
    """
    print(f"\n  [Strategy 3: TIME-BASED CLOSURE]")

    top_20_symbols = {opp['symbol'] for opp in top_20_opportunities}
    top_50_symbols = {opp['symbol'] for opp in top_20_opportunities[:50]} if len(top_20_opportunities) >= 50 else top_20_symbols

    closed_count = 0

    for position in list(position_manager.get_all_open()):
        if position.symbol in top_20_symbols:
            continue

        # Текущий PnL
        current_price = exchange.get_current_price(position.symbol)
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        # Возраст позиции
        position_age = datetime.now() - position.entry_time
        age_hours = position_age.total_seconds() / 3600

        # ПРАВИЛА ЗАКРЫТИЯ:

        # 1. Вне TOP-20 и старше 24 часов
        if age_hours > 24:
            print(f"  ⏰ Closing {position.symbol}: Out TOP-20 + {age_hours:.1f}h old")
            position_manager.close_position(position.symbol, reason='MANUAL_TIME_24H')
            closed_count += 1
            continue

        # 2. Вне TOP-50 и старше 12 часов
        if position.symbol not in top_50_symbols and age_hours > 12:
            print(f"  📉 Closing {position.symbol}: Out TOP-50 + {age_hours:.1f}h old")
            position_manager.close_position(position.symbol, reason='MANUAL_TIME_12H')
            closed_count += 1
            continue

        # 3. Критический убыток (близко к SL)
        if pnl_pct < -7.0:
            print(f"  🚨 Closing {position.symbol}: Critical loss {pnl_pct:.2f}%")
            position_manager.close_position(position.symbol, reason='MANUAL_CRITICAL_LOSS')
            closed_count += 1
            continue

        # Молодая позиция - даем шанс
        print(f"  ⏳ Keeping {position.symbol}: Young position ({age_hours:.1f}h)")

    return closed_count


# ============================================================================
# СТРАТЕГИЯ 4: "Градация приоритетов" (PRIORITY-BASED)
# ============================================================================

def strategy_4_priority_based(position_manager, top_20_opportunities, exchange, haiku_analyzer=None):
    """
    Рассчитываем score для каждой позиции, закрываем только слабые

    Ожидаемое улучшение: +3.0% до +4.0%
    """
    print(f"\n  [Strategy 4: PRIORITY-BASED CLOSURE]")

    top_20_symbols = {opp['symbol'] for opp in top_20_opportunities}

    # Создаем словарь EV для быстрого поиска
    ev_dict = {opp['symbol']: opp.get('ev', 1.0) for opp in top_20_opportunities}

    closed_count = 0

    for position in list(position_manager.get_all_open()):
        if position.symbol in top_20_symbols:
            continue

        # Текущий PnL
        current_price = exchange.get_current_price(position.symbol)
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        # Возраст позиции
        position_age = datetime.now() - position.entry_time
        age_hours = position_age.total_seconds() / 3600

        # РАСЧЕТ SCORE
        score = 0

        # Факторы ЗА удержание:
        if pnl_pct > 0:
            score += 50  # В прибыли

        current_ev = ev_dict.get(position.symbol, 1.0)
        if current_ev > 1.5:
            score += 30  # Высокая EV

        if haiku_analyzer:
            analysis = haiku_analyzer.get_cached_analysis(position.symbol)
            if analysis and analysis.get('fundamental_score', 0.5) > 0.6:
                score += 20  # Хороший фундаментал

        # Факторы ПРОТИВ удержания:
        if pnl_pct < -3.0:
            score -= 30  # Убыток

        if age_hours > 36:
            score -= 20  # Старая

        # Проверяем, насколько далеко из топа
        position_rank = None
        for i, opp in enumerate(top_20_opportunities):
            if opp['symbol'] == position.symbol:
                position_rank = i + 1
                break

        if position_rank is None or position_rank > 50:
            score -= 40  # Вне TOP-50
        elif position_rank > 20:
            score -= 10  # Вне TOP-20 но в TOP-50

        print(f"  📊 {position.symbol}: score={score} (pnl={pnl_pct:.2f}%, age={age_hours:.1f}h, rank={position_rank})")

        # Закрываем только если score отрицательный
        if score < 0:
            print(f"    ❌ Closing {position.symbol}: Low priority (score={score})")
            position_manager.close_position(position.symbol, reason='MANUAL_LOW_PRIORITY')
            closed_count += 1
        else:
            print(f"    ✅ Keeping {position.symbol}: High priority (score={score})")

    return closed_count


# ============================================================================
# СТРАТЕГИЯ 5: "Защита прибыльных" (PROFIT PROTECTION) ⭐ RECOMMENDED
# ============================================================================

def strategy_5_profit_protection(position_manager, top_20_opportunities, exchange):
    """
    Никогда не закрывать прибыльные позиции.
    Быстро избавляться от убыточных.

    Ожидаемое улучшение: +3.2% до +3.8%
    ⭐ РЕКОМЕНДУЕТСЯ для начала
    """
    print(f"\n  [Strategy 5: PROFIT PROTECTION] ⭐")

    top_20_symbols = {opp['symbol'] for opp in top_20_opportunities}

    closed_count = 0

    for position in list(position_manager.get_all_open()):
        if position.symbol in top_20_symbols:
            continue

        # Текущий PnL
        current_price = exchange.get_current_price(position.symbol)
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        # Возраст позиции
        position_age = datetime.now() - position.entry_time
        age_hours = position_age.total_seconds() / 3600

        # ГЛАВНОЕ ПРАВИЛО: НЕ трогать прибыльные!
        if pnl_pct > 0:
            print(f"  💰 Keeping {position.symbol}: In profit +{pnl_pct:.2f}% (let trailing stop work)")
            continue

        # Для убыточных - применяем строгие критерии:

        # 1. Средний убыток + достаточно времени
        if pnl_pct < -3.0 and age_hours > 12:
            print(f"  ❌ Closing {position.symbol}: Loss {pnl_pct:.2f}% + {age_hours:.1f}h old")
            position_manager.close_position(position.symbol, reason='MANUAL_LOSS_TIME')
            closed_count += 1
            continue

        # 2. Большой убыток - закрываем сразу
        if pnl_pct < -5.0:
            print(f"  🚨 Closing {position.symbol}: Heavy loss {pnl_pct:.2f}%")
            position_manager.close_position(position.symbol, reason='MANUAL_HEAVY_LOSS')
            closed_count += 1
            continue

        # Мелкий убыток и молодая - даем шанс
        print(f"  ⏳ Keeping {position.symbol}: Small loss {pnl_pct:.2f}%, young ({age_hours:.1f}h)")

    return closed_count


# ============================================================================
# ИНТЕГРАЦИЯ В ОСНОВНОЙ КОД
# ============================================================================

def integrate_strategy_into_bot(strategy_number=5):
    """
    Пример интеграции стратегии в binance_bot_main.py

    Заменить строки 915-919 на:
    """

    example_code = """
    # ════════════════════════════════════════════════════════════════════════
    # УПРАВЛЕНИЕ ПОЗИЦИЯМИ ПРИ РЕБАЛАНСИРОВКЕ
    # Стратегия {}: {}
    # ════════════════════════════════════════════════════════════════════════

    print(f"\\n  Checking if existing positions should be closed (using top-20)...", flush=True)

    positions_before_close = len(self.position_manager.get_all_open())

    # Выбранная стратегия
    {}

    # Логирование
    print(f"  Closed: {{closed_count}} positions")
    """.format(
        strategy_number,
        {
            1: "NO CLOSURE - Позиции закрываются только по TP/SL",
            2: "SMART CLOSURE - Закрытие только при объективных причинах",
            3: "TIME-BASED - Временная толерантность",
            4: "PRIORITY-BASED - Градация по приоритетам",
            5: "PROFIT PROTECTION - Защита прибыльных позиций ⭐"
        }[strategy_number],
        {
            1: "closed_count = 0  # No manual closure",
            2: "closed_count = self._strategy_smart_closure(top_20_opportunities)",
            3: "closed_count = self._strategy_time_based(top_20_opportunities)",
            4: "closed_count = self._strategy_priority_based(top_20_opportunities)",
            5: "closed_count = self._strategy_profit_protection(top_20_opportunities)"
        }[strategy_number]
    )

    return example_code


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🧪 ТЕСТИРОВАНИЕ СТРАТЕГИЙ УПРАВЛЕНИЯ ПОЗИЦИЯМИ")
    print("="*80)

    # Примеры интеграции
    for i in range(1, 6):
        print(f"\n{'='*80}")
        print(f"СТРАТЕГИЯ {i}")
        print("="*80)
        print(integrate_strategy_into_bot(i))

    print("\n" + "="*80)
    print("✅ Выберите стратегию и интегрируйте в binance_bot_main.py")
    print("="*80)
