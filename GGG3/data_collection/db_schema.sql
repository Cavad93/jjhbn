-- ============================================================================
-- СХЕМА БАЗЫ ДАННЫХ ДЛЯ BINANCE BOT
-- PostgreSQL 12+
-- ============================================================================

-- Создание базы данных (выполнить от суперпользователя)
-- CREATE DATABASE binance_bot OWNER bot_user;

-- Подключиться к базе: \c binance_bot

-- ============================================================================
-- РАСШИРЕНИЯ
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Для быстрого поиска по тексту

-- ============================================================================
-- ТАБЛИЦА: ohlcv_data
-- Хранение OHLCV данных для всех таймфреймов
-- ============================================================================

CREATE TABLE IF NOT EXISTS ohlcv_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,                -- BTCUSDT, ETHUSDT, etc.
    timeframe VARCHAR(10) NOT NULL,             -- 5m, 15m, 30m, 4h
    timestamp TIMESTAMPTZ NOT NULL,             -- Временная метка свечи
    open DECIMAL(20, 8) NOT NULL,               -- Цена открытия
    high DECIMAL(20, 8) NOT NULL,               -- Максимальная цена
    low DECIMAL(20, 8) NOT NULL,                -- Минимальная цена
    close DECIMAL(20, 8) NOT NULL,              -- Цена закрытия
    volume DECIMAL(20, 8) NOT NULL,             -- Объём торгов
    quote_volume DECIMAL(20, 8),                -- Объём в USDT (опционально)
    trades_count INTEGER,                       -- Количество сделок (опционально)
    created_at TIMESTAMPTZ DEFAULT NOW(),       -- Время добавления в БД
    updated_at TIMESTAMPTZ DEFAULT NOW(),       -- Время последнего обновления

    -- Уникальность: одна свеча для комбинации symbol+timeframe+timestamp
    CONSTRAINT ohlcv_unique UNIQUE (symbol, timeframe, timestamp)
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data(symbol);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv_data(timeframe);
CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_data(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp ON ohlcv_data(symbol, timeframe, timestamp DESC);

-- Партиционирование по месяцам для ускорения (опционально, для больших объёмов)
-- CREATE TABLE ohlcv_data_2025_11 PARTITION OF ohlcv_data
--     FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- Комментарии
COMMENT ON TABLE ohlcv_data IS 'OHLCV данные для всех торгуемых пар и таймфреймов';
COMMENT ON COLUMN ohlcv_data.symbol IS 'Символ торговой пары (BTCUSDT)';
COMMENT ON COLUMN ohlcv_data.timeframe IS 'Таймфрейм свечи (5m, 15m, 30m, 4h)';
COMMENT ON COLUMN ohlcv_data.timestamp IS 'Временная метка открытия свечи (UTC)';

-- ============================================================================
-- ТАБЛИЦА: active_symbols
-- Список активных символов (топ-10 по EV)
-- ============================================================================

CREATE TABLE IF NOT EXISTS active_symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,         -- BTCUSDT
    ev_score DECIMAL(10, 6),                    -- Expected Value (EV)
    rank INTEGER,                               -- Ранг по EV (1-10)
    volume_24h DECIMAL(20, 2),                  -- Объём торгов за 24ч
    last_position_at TIMESTAMPTZ,               -- Дата последней открытой позиции
    positions_count INTEGER DEFAULT 0,          -- Количество открытых позиций
    is_active BOOLEAN DEFAULT TRUE,             -- Активен ли символ
    added_at TIMESTAMPTZ DEFAULT NOW(),         -- Когда добавлен в топ-10
    updated_at TIMESTAMPTZ DEFAULT NOW(),       -- Последнее обновление

    CHECK (rank >= 1 AND rank <= 10)            -- Только топ-10
);

-- Индексы
CREATE INDEX IF NOT EXISTS idx_active_symbols_rank ON active_symbols(rank);
CREATE INDEX IF NOT EXISTS idx_active_symbols_is_active ON active_symbols(is_active);
CREATE INDEX IF NOT EXISTS idx_active_symbols_last_position ON active_symbols(last_position_at);

COMMENT ON TABLE active_symbols IS 'Список активных символов (топ-10 по EV)';
COMMENT ON COLUMN active_symbols.ev_score IS 'Ожидаемая прибыль (Expected Value)';
COMMENT ON COLUMN active_symbols.rank IS 'Ранг по EV (1-10)';
COMMENT ON COLUMN active_symbols.last_position_at IS 'Дата последней позиции для отслеживания очистки';

-- ============================================================================
-- ТАБЛИЦА: data_stats
-- Статистика по загруженным данным
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_stats (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    rows_count INTEGER DEFAULT 0,               -- Количество свечей
    first_timestamp TIMESTAMPTZ,                -- Первая временная метка
    last_timestamp TIMESTAMPTZ,                 -- Последняя временная метка
    data_size_mb DECIMAL(10, 2),                -- Размер данных (MB)
    last_update TIMESTAMPTZ DEFAULT NOW(),      -- Последнее обновление

    CONSTRAINT data_stats_unique UNIQUE (symbol, timeframe)
);

-- Индексы
CREATE INDEX IF NOT EXISTS idx_data_stats_symbol ON data_stats(symbol);
CREATE INDEX IF NOT EXISTS idx_data_stats_last_update ON data_stats(last_update DESC);

COMMENT ON TABLE data_stats IS 'Статистика по загруженным данным для каждого символа';

-- ============================================================================
-- ТАБЛИЦА: cleanup_log
-- Лог удаления старых данных
-- ============================================================================

CREATE TABLE IF NOT EXISTS cleanup_log (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframes TEXT[],                          -- Массив удалённых таймфреймов
    rows_deleted INTEGER,                       -- Количество удалённых строк
    reason VARCHAR(100),                        -- Причина удаления
    deleted_at TIMESTAMPTZ DEFAULT NOW()        -- Время удаления
);

-- Индекс
CREATE INDEX IF NOT EXISTS idx_cleanup_log_deleted_at ON cleanup_log(deleted_at DESC);

COMMENT ON TABLE cleanup_log IS 'Журнал очистки старых данных';
COMMENT ON COLUMN cleanup_log.reason IS 'Причина удаления (not_in_top10, no_positions_7days, manual)';

-- ============================================================================
-- ФУНКЦИИ
-- ============================================================================

-- Функция для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггеры для автоматического обновления updated_at
CREATE TRIGGER update_ohlcv_data_updated_at
    BEFORE UPDATE ON ohlcv_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_active_symbols_updated_at
    BEFORE UPDATE ON active_symbols
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_stats_updated_at
    BEFORE UPDATE ON data_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ПРЕДСТАВЛЕНИЯ (VIEWS)
-- ============================================================================

-- Представление: последние данные по каждому символу и таймфрейму
CREATE OR REPLACE VIEW latest_ohlcv AS
SELECT DISTINCT ON (symbol, timeframe)
    symbol,
    timeframe,
    timestamp,
    open,
    high,
    low,
    close,
    volume
FROM ohlcv_data
ORDER BY symbol, timeframe, timestamp DESC;

COMMENT ON VIEW latest_ohlcv IS 'Последние OHLCV данные для каждого символа и таймфрейма';

-- Представление: статистика по активным символам
CREATE OR REPLACE VIEW active_symbols_stats AS
SELECT
    a.symbol,
    a.rank,
    a.ev_score,
    a.positions_count,
    a.last_position_at,
    d.rows_count AS ohlcv_rows,
    d.last_update AS data_last_update,
    EXTRACT(EPOCH FROM (NOW() - a.last_position_at))/86400 AS days_since_last_position
FROM active_symbols a
LEFT JOIN data_stats d ON a.symbol = d.symbol AND d.timeframe = '4h'
WHERE a.is_active = TRUE
ORDER BY a.rank;

COMMENT ON VIEW active_symbols_stats IS 'Статистика по активным символам с данными';

-- ============================================================================
-- ФУНКЦИЯ: Получение OHLCV данных
-- ============================================================================

CREATE OR REPLACE FUNCTION get_ohlcv(
    p_symbol VARCHAR(20),
    p_timeframe VARCHAR(10),
    p_limit INTEGER DEFAULT 500
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        o.timestamp,
        o.open,
        o.high,
        o.low,
        o.close,
        o.volume
    FROM ohlcv_data o
    WHERE o.symbol = p_symbol
      AND o.timeframe = p_timeframe
    ORDER BY o.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_ohlcv IS 'Получение последних N свечей для символа и таймфрейма';

-- ============================================================================
-- ФУНКЦИЯ: Удаление старых данных для символа
-- ============================================================================

CREATE OR REPLACE FUNCTION cleanup_symbol_data(
    p_symbol VARCHAR(20),
    p_reason VARCHAR(100) DEFAULT 'manual'
)
RETURNS INTEGER AS $$
DECLARE
    v_rows_deleted INTEGER := 0;
    v_timeframes TEXT[];
BEGIN
    -- Получаем список таймфреймов для символа
    SELECT ARRAY_AGG(DISTINCT timeframe) INTO v_timeframes
    FROM ohlcv_data
    WHERE symbol = p_symbol;

    -- Удаляем OHLCV данные
    DELETE FROM ohlcv_data
    WHERE symbol = p_symbol;

    GET DIAGNOSTICS v_rows_deleted = ROW_COUNT;

    -- Удаляем статистику
    DELETE FROM data_stats
    WHERE symbol = p_symbol;

    -- Деактивируем символ в active_symbols
    UPDATE active_symbols
    SET is_active = FALSE
    WHERE symbol = p_symbol;

    -- Записываем в лог
    INSERT INTO cleanup_log (symbol, timeframes, rows_deleted, reason)
    VALUES (p_symbol, v_timeframes, v_rows_deleted, p_reason);

    RETURN v_rows_deleted;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_symbol_data IS 'Удаление всех данных для символа с логированием';

-- ============================================================================
-- ФУНКЦИЯ: Обновление статистики
-- ============================================================================

CREATE OR REPLACE FUNCTION refresh_data_stats()
RETURNS VOID AS $$
BEGIN
    -- Обновляем статистику для всех символов и таймфреймов
    INSERT INTO data_stats (symbol, timeframe, rows_count, first_timestamp, last_timestamp, data_size_mb)
    SELECT
        symbol,
        timeframe,
        COUNT(*) AS rows_count,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp,
        ROUND(pg_column_size(ohlcv_data.*) * COUNT(*) / 1024.0 / 1024.0, 2) AS data_size_mb
    FROM ohlcv_data
    GROUP BY symbol, timeframe
    ON CONFLICT (symbol, timeframe)
    DO UPDATE SET
        rows_count = EXCLUDED.rows_count,
        first_timestamp = EXCLUDED.first_timestamp,
        last_timestamp = EXCLUDED.last_timestamp,
        data_size_mb = EXCLUDED.data_size_mb,
        last_update = NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_data_stats IS 'Обновление статистики по всем данным';

-- ============================================================================
-- ПРАВА ДОСТУПА
-- ============================================================================

-- Создание пользователя для бота (если не существует)
-- CREATE USER bot_user WITH PASSWORD 'secure_password_here';

-- Выдача прав
-- GRANT ALL PRIVILEGES ON DATABASE binance_bot TO bot_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bot_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bot_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO bot_user;

-- ============================================================================
-- ТЕСТОВЫЕ ДАННЫЕ (опционально)
-- ============================================================================

-- Вставка тестовых данных
-- INSERT INTO active_symbols (symbol, ev_score, rank, volume_24h)
-- VALUES
--     ('BTCUSDT', 0.045, 1, 25000000000),
--     ('ETHUSDT', 0.042, 2, 15000000000),
--     ('BNBUSDT', 0.038, 3, 8000000000);

-- ============================================================================
-- КОНЕЦ СХЕМЫ
-- ============================================================================
