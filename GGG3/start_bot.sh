#!/bin/bash
################################################################################
# Скрипт запуска Binance Trading Bot
# Версия: 1.0
################################################################################

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Директория проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# PID файл
PID_FILE="$PROJECT_DIR/.bot.pid"
LOG_FILE="$PROJECT_DIR/logs/binance_bot.log"

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
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Проверка, что бот не запущен
check_not_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            print_error "Бот уже запущен (PID: $PID)"
            echo ""
            echo "Для остановки используйте:"
            echo "  ./stop_bot.sh"
            echo ""
            echo "Для проверки статуса:"
            echo "  ./status_bot.sh"
            exit 1
        else
            # PID файл существует, но процесс не запущен - очищаем
            rm -f "$PID_FILE"
        fi
    fi
}

# Проверка виртуального окружения
check_virtualenv() {
    if [ ! -d "venv" ]; then
        print_error "Виртуальное окружение не найдено!"
        echo "Запустите установку: ./install.sh"
        exit 1
    fi
}

# Проверка .env файла
check_env_file() {
    if [ ! -f ".env" ]; then
        print_error ".env файл не найден!"
        echo "Создайте .env файл:"
        echo "  cp .env.example .env"
        echo "  nano .env"
        exit 1
    fi
}

# Создание директорий для логов
create_log_dirs() {
    mkdir -p logs
    mkdir -p data/positions
}

# Запуск в интерактивном режиме
start_interactive() {
    print_header "Запуск бота (интерактивный режим)"

    # Определение режима
    MODE="paper"
    if [ "$1" == "--live" ]; then
        MODE="live"
        print_warning "LIVE TRADING РЕЖИМ!"
        print_warning "Будут использованы РЕАЛЬНЫЕ деньги!"
        echo ""
        read -p "Вы уверены? Введите 'YES' для продолжения: " CONFIRM
        if [ "$CONFIRM" != "YES" ]; then
            print_info "Запуск отменён"
            exit 0
        fi
    else
        print_info "PAPER TRADING режим"
    fi

    echo ""
    print_info "Активация виртуального окружения..."
    source venv/bin/activate

    print_info "Запуск бота..."
    if [ "$MODE" == "live" ]; then
        python3 binance_bot_main.py --live
    else
        python3 binance_bot_main.py --paper
    fi
}

# Запуск в фоновом режиме
start_background() {
    print_header "Запуск бота (фоновый режим)"

    # Определение режима
    MODE="paper"
    if [ "$1" == "--live" ]; then
        MODE="live"
        print_warning "LIVE TRADING РЕЖИМ!"
        print_warning "Будут использованы РЕАЛЬНЫЕ деньги!"
        echo ""
        read -p "Вы уверены? Введите 'YES' для продолжения: " CONFIRM
        if [ "$CONFIRM" != "YES" ]; then
            print_info "Запуск отменён"
            exit 0
        fi
    else
        print_info "PAPER TRADING режим"
    fi

    echo ""
    print_info "Активация виртуального окружения..."
    source venv/bin/activate

    print_info "Запуск бота в фоновом режиме..."

    # Запуск с nohup
    if [ "$MODE" == "live" ]; then
        nohup python3 binance_bot_main.py --live > "$LOG_FILE" 2>&1 &
    else
        nohup python3 binance_bot_main.py --paper > "$LOG_FILE" 2>&1 &
    fi

    # Сохраняем PID
    echo $! > "$PID_FILE"
    PID=$(cat "$PID_FILE")

    # Ждём немного чтобы убедиться что процесс запустился
    sleep 2

    if ps -p "$PID" > /dev/null 2>&1; then
        print_success "Бот запущен в фоновом режиме (PID: $PID)"
        echo ""
        echo "Для просмотра логов:"
        echo "  tail -f $LOG_FILE"
        echo ""
        echo "Для остановки:"
        echo "  ./stop_bot.sh"
        echo ""
        echo "Для проверки статуса:"
        echo "  ./status_bot.sh"
    else
        print_error "Не удалось запустить бота!"
        rm -f "$PID_FILE"
        echo ""
        echo "Проверьте логи:"
        echo "  cat $LOG_FILE"
        exit 1
    fi
}

# Главная функция
main() {
    # Проверки
    check_not_running
    check_virtualenv
    check_env_file
    create_log_dirs

    # Определение режима запуска
    BACKGROUND=false
    MODE_ARG=""

    for arg in "$@"; do
        case $arg in
            --background|-b)
                BACKGROUND=true
                shift
                ;;
            --live)
                MODE_ARG="--live"
                shift
                ;;
            --paper)
                MODE_ARG="--paper"
                shift
                ;;
            --help|-h)
                echo "Использование: $0 [OPTIONS]"
                echo ""
                echo "Опции:"
                echo "  --paper          Запуск в paper trading режиме (по умолчанию)"
                echo "  --live           Запуск в live trading режиме (ОСТОРОЖНО!)"
                echo "  --background, -b Запуск в фоновом режиме"
                echo "  --help, -h       Показать эту справку"
                echo ""
                echo "Примеры:"
                echo "  $0                    # Paper trading, интерактивный режим"
                echo "  $0 --background       # Paper trading, фоновый режим"
                echo "  $0 --live             # Live trading, интерактивный режим"
                echo "  $0 --live --background # Live trading, фоновый режим"
                exit 0
                ;;
        esac
    done

    # Запуск
    if [ "$BACKGROUND" = true ]; then
        start_background "$MODE_ARG"
    else
        start_interactive "$MODE_ARG"
    fi
}

main "$@"
