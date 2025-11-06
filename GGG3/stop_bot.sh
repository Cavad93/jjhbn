#!/bin/bash
################################################################################
# Скрипт остановки Binance Trading Bot
# Версия: 1.0
################################################################################

set -e

# Цвета
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Директория проекта
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# PID файл
PID_FILE="$PROJECT_DIR/.bot.pid"

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Проверка, что бот запущен
if [ ! -f "$PID_FILE" ]; then
    print_error "Бот не запущен (PID файл не найден)"
    exit 1
fi

PID=$(cat "$PID_FILE")

# Проверка существования процесса
if ! ps -p "$PID" > /dev/null 2>&1; then
    print_error "Процесс с PID $PID не найден"
    rm -f "$PID_FILE"
    exit 1
fi

# Остановка бота
print_info "Остановка бота (PID: $PID)..."

# Сначала пробуем graceful shutdown (SIGTERM)
kill -TERM "$PID" 2>/dev/null || true

# Ждём до 10 секунд
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        print_success "Бот остановлен успешно"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Если не остановился, используем SIGKILL
print_info "Процесс не остановился, используем принудительную остановку..."
kill -KILL "$PID" 2>/dev/null || true

sleep 1

if ! ps -p "$PID" > /dev/null 2>&1; then
    print_success "Бот остановлен принудительно"
    rm -f "$PID_FILE"
    exit 0
else
    print_error "Не удалось остановить бота!"
    exit 1
fi
