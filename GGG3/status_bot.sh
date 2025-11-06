#!/bin/bash
################################################################################
# Скрипт проверки статуса Binance Trading Bot
# Версия: 1.0
################################################################################

# Цвета
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Директория проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Файлы
PID_FILE="$PROJECT_DIR/.bot.pid"
LOG_FILE="$PROJECT_DIR/logs/binance_bot.log"
POSITIONS_FILE="$PROJECT_DIR/data/positions/positions.json"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${CYAN}$1${NC}"
}

# Проверка статуса процесса
check_process_status() {
    print_header "Статус процесса"

    if [ ! -f "$PID_FILE" ]; then
        print_error "Бот НЕ ЗАПУЩЕН (PID файл не найден)"
        return 1
    fi

    PID=$(cat "$PID_FILE")

    if ps -p "$PID" > /dev/null 2>&1; then
        print_success "Бот РАБОТАЕТ (PID: $PID)"

        # Информация о процессе
        echo ""
        print_info "Информация о процессе:"
        ps -p "$PID" -o pid,ppid,user,%cpu,%mem,vsz,rss,etime,cmd --no-headers | \
            awk '{printf "  PID:        %s\n  Родитель:   %s\n  Пользователь: %s\n  CPU:        %s%%\n  Память:     %s%% (%.1f MB)\n  Uptime:     %s\n", $1, $2, $3, $4, $5, $7/1024, $8}'

        return 0
    else
        print_error "Бот НЕ РАБОТАЕТ (процесс PID $PID не найден)"
        echo ""
        print_info "PID файл существует, но процесс не запущен"
        print_info "Возможно бот был остановлен некорректно"
        print_info "Очистите PID файл: rm $PID_FILE"
        return 1
    fi
}

# Проверка последних логов
check_logs() {
    print_header "Последние логи"

    if [ ! -f "$LOG_FILE" ]; then
        print_warning "Лог файл не найден: $LOG_FILE"
        return 1
    fi

    # Показываем последние 15 строк
    print_info "Последние 15 строк из лога:"
    echo ""
    tail -n 15 "$LOG_FILE" | while IFS= read -r line; do
        # Подсветка важных сообщений
        if [[ $line == *"ERROR"* ]] || [[ $line == *"❌"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ $line == *"WARNING"* ]] || [[ $line == *"⚠️"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ $line == *"SUCCESS"* ]] || [[ $line == *"✅"* ]]; then
            echo -e "${GREEN}$line${NC}"
        else
            echo "$line"
        fi
    done

    echo ""
    print_info "Для просмотра всех логов: tail -f $LOG_FILE"
}

# Проверка открытых позиций
check_positions() {
    print_header "Открытые позиции"

    if [ ! -f "$POSITIONS_FILE" ]; then
        print_warning "Файл позиций не найден: $POSITIONS_FILE"
        return 1
    fi

    # Проверяем, установлен ли jq для парсинга JSON
    if command -v jq &> /dev/null; then
        OPEN_COUNT=$(jq '.open_positions | length' "$POSITIONS_FILE" 2>/dev/null || echo "0")
        CLOSED_COUNT=$(jq '.closed_positions | length' "$POSITIONS_FILE" 2>/dev/null || echo "0")
        TOTAL_CLOSED=$(jq '.metadata.total_closed // 0' "$POSITIONS_FILE" 2>/dev/null || echo "0")

        print_info "Открыто позиций:  $OPEN_COUNT"
        print_info "Закрыто (последние): $CLOSED_COUNT"
        print_info "Закрыто всего:    $TOTAL_CLOSED"

        if [ "$OPEN_COUNT" -gt 0 ]; then
            echo ""
            print_info "Список открытых позиций:"
            jq -r '.open_positions[] | "  - \(.symbol) \(.direction) @ $\(.entry_price) (TP: $\(.tp_price), SL: $\(.sl_price))"' "$POSITIONS_FILE" 2>/dev/null || echo "  Ошибка чтения позиций"
        fi

        # Статистика по последним закрытым позициям
        if [ "$CLOSED_COUNT" -gt 0 ]; then
            echo ""
            print_info "Последние 5 закрытых позиций:"
            jq -r '.closed_positions[-5:] | .[] | "  - \(.symbol) \(.direction): PnL=\(.pnl_pct)% (\(.exit_reason))"' "$POSITIONS_FILE" 2>/dev/null || echo "  Ошибка чтения позиций"
        fi
    else
        # Fallback без jq
        print_info "Сырые данные позиций (установите jq для красивого вывода):"
        echo ""
        cat "$POSITIONS_FILE" | python3 -m json.tool 2>/dev/null | head -n 30 || cat "$POSITIONS_FILE" | head -n 30
        echo ""
        print_info "Для форматированного вывода установите: sudo apt-get install jq"
    fi
}

# Проверка использования ресурсов
check_resources() {
    print_header "Использование ресурсов"

    # CPU и память системы
    print_info "Система:"
    echo ""

    # CPU Load
    if [ -f /proc/loadavg ]; then
        LOAD=$(cat /proc/loadavg | awk '{print $1, $2, $3}')
        print_info "  Load Average (1m, 5m, 15m): $LOAD"
    fi

    # Память
    if command -v free &> /dev/null; then
        MEMORY=$(free -h | awk '/^Mem:/ {printf "  RAM: %s / %s используется (%.0f%%)\n", $3, $2, ($3/$2)*100}')
        print_info "$MEMORY"
    fi

    # Диск
    if [ -d "$PROJECT_DIR" ]; then
        DISK=$(df -h "$PROJECT_DIR" | awk 'NR==2 {printf "  Диск: %s / %s используется (%s)\n", $3, $2, $5}')
        print_info "$DISK"
    fi

    echo ""

    # Ресурсы бота
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_info "Бот:"
            echo ""
            ps -p "$PID" -o %cpu,%mem,vsz,rss --no-headers | \
                awk '{printf "  CPU: %s%%\n  RAM: %s%% (%.1f MB)\n", $1, $2, $4/1024}'

            # Количество потоков
            THREADS=$(ps -p "$PID" -o nlwp --no-headers 2>/dev/null || echo "N/A")
            print_info "  Потоков: $THREADS"

            # Открытые файлы
            if command -v lsof &> /dev/null; then
                FD_COUNT=$(lsof -p "$PID" 2>/dev/null | wc -l)
                print_info "  Открытых файлов: $FD_COUNT"
            fi
        fi
    fi
}

# Проверка конфигурации
check_configuration() {
    print_header "Конфигурация"

    if [ -f ".env" ]; then
        print_success ".env файл найден"

        # Проверка ключевых переменных (без вывода значений)
        print_info "Настроенные переменные:"

        if grep -q "^BINANCE_API_KEY=" .env && [ -n "$(grep "^BINANCE_API_KEY=" .env | cut -d= -f2)" ]; then
            print_info "  ✓ BINANCE_API_KEY"
        else
            print_warning "  ✗ BINANCE_API_KEY (не настроен)"
        fi

        if grep -q "^BINANCE_SECRET_KEY=" .env && [ -n "$(grep "^BINANCE_SECRET_KEY=" .env | cut -d= -f2)" ]; then
            print_info "  ✓ BINANCE_SECRET_KEY"
        else
            print_warning "  ✗ BINANCE_SECRET_KEY (не настроен)"
        fi

        if grep -q "^TELEGRAM_BOT_TOKEN=" .env && [ -n "$(grep "^TELEGRAM_BOT_TOKEN=" .env | cut -d= -f2)" ]; then
            print_info "  ✓ TELEGRAM_BOT_TOKEN"
        else
            print_warning "  ✗ TELEGRAM_BOT_TOKEN (не настроен)"
        fi

        # Режим
        MODE=$(grep "^PAPER_TRADING_MODE=" .env | cut -d= -f2 || echo "true")
        if [ "$MODE" == "true" ]; then
            print_info "  Режим: PAPER TRADING ✓"
        else
            print_warning "  Режим: LIVE TRADING ⚠️"
        fi
    else
        print_error ".env файл не найден!"
    fi
}

# Главная функция
main() {
    clear

    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   BINANCE TRADING BOT - СТАТУС         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""

    # Проверки
    BOT_RUNNING=false
    if check_process_status; then
        BOT_RUNNING=true
    fi

    echo ""
    check_configuration

    if [ "$BOT_RUNNING" = true ]; then
        echo ""
        check_resources

        echo ""
        check_positions

        echo ""
        check_logs
    fi

    echo ""
    print_header "Быстрые команды"
    echo ""

    if [ "$BOT_RUNNING" = true ]; then
        print_info "Остановить бота:     ./stop_bot.sh"
        print_info "Просмотр логов:      tail -f $LOG_FILE"
    else
        print_info "Запустить бота:      ./start_bot.sh"
        print_info "Запуск в фоне:       ./start_bot.sh --background"
    fi

    print_info "Обновить статус:     ./status_bot.sh"

    echo ""
}

main "$@"
