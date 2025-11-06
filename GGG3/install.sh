#!/bin/bash
################################################################################
# Автоматическая установка Binance Trading Bot
# Версия: 1.0
# Автор: Claude Code
################################################################################

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции вывода
print_header() {
    echo -e "${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Проверка версии Python
check_python() {
    print_header "Проверка Python"

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 не найден!"
        echo "Установите Python 3.9 или выше:"
        echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
        echo "  macOS: brew install python3"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
        print_error "Python версия $PYTHON_VERSION слишком старая!"
        echo "Требуется Python 3.9 или выше"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION ✓"
}

# Очистка старого окружения
clean_old_environment() {
    print_header "Очистка старого окружения"

    # Деактивация текущего окружения
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "Деактивация текущего виртуального окружения..."
        deactivate 2>/dev/null || true
    fi

    # Удаление старых виртуальных окружений
    for venv_dir in venv env .venv; do
        if [ -d "$venv_dir" ]; then
            print_info "Удаление $venv_dir/..."
            rm -rf "$venv_dir"
        fi
    done

    # Очистка кэша Python
    print_info "Очистка кэша Python..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

    print_success "Старое окружение очищено"
}

# Создание виртуального окружения
create_virtualenv() {
    print_header "Создание виртуального окружения"

    print_info "Создание venv..."
    python3 -m venv venv

    print_success "Виртуальное окружение создано"
}

# Установка зависимостей
install_dependencies() {
    print_header "Установка зависимостей"

    print_info "Активация виртуального окружения..."
    source venv/bin/activate

    print_info "Обновление pip, setuptools, wheel..."
    python3 -m pip install --upgrade pip setuptools wheel --quiet

    print_info "Установка зависимостей из requirements.txt..."
    print_warning "Это может занять 5-10 минут, пожалуйста подождите..."

    pip3 install -r requirements.txt

    print_success "Все зависимости установлены"
}

# Проверка установки
verify_installation() {
    print_header "Проверка установки"

    source venv/bin/activate

    if [ -f "verify_dependencies.py" ]; then
        print_info "Запуск проверки зависимостей..."
        python3 verify_dependencies.py
    else
        print_warning "Файл verify_dependencies.py не найден, пропускаем проверку"
    fi
}

# Создание директорий
create_directories() {
    print_header "Создание необходимых директорий"

    DIRS=(
        "data/positions"
        "data/market_data"
        "logs"
        "models/saved"
    )

    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            print_info "Создание $dir/..."
            mkdir -p "$dir"
        fi
    done

    # Создание .gitkeep для пустых директорий
    touch models/saved/.gitkeep

    print_success "Директории созданы"
}

# Настройка .env файла
setup_env_file() {
    print_header "Настройка конфигурации"

    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_info "Создание .env из .env.example..."
            cp .env.example .env
            print_success ".env файл создан"
            print_warning "Не забудьте отредактировать .env файл и добавить ваши API ключи!"
            echo ""
            echo "Откройте файл: nano .env"
            echo "Или: vim .env"
        else
            print_error ".env.example не найден!"
        fi
    else
        print_info ".env файл уже существует, пропускаем..."
    fi
}

# Проверка прав доступа
check_permissions() {
    print_header "Проверка прав доступа"

    # Проверка прав на запись
    if [ ! -w "." ]; then
        print_error "Нет прав на запись в текущую директорию!"
        echo "Выполните: sudo chown -R $USER:$USER $(pwd)"
        exit 1
    fi

    # Установка правильных прав
    chmod -R 755 data/ logs/ models/ 2>/dev/null || true

    print_success "Права доступа настроены"
}

# Установка TA-Lib (опционально)
install_talib() {
    print_header "Установка TA-Lib (опционально)"

    # Проверяем, установлен ли TA-Lib
    if ldconfig -p | grep -q libta-lib; then
        print_success "TA-Lib уже установлен"
        return 0
    fi

    print_warning "TA-Lib не найден"

    # Автоматическая установка только на Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        read -p "Установить TA-Lib автоматически? (требуются права sudo) [y/N]: " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Установка TA-Lib..."

            # Проверка наличия sudo
            if ! command -v sudo &> /dev/null; then
                print_error "sudo не найден, установите TA-Lib вручную"
                return 1
            fi

            # Установка зависимостей для сборки
            sudo apt-get update -qq
            sudo apt-get install -y build-essential wget

            # Скачивание и установка TA-Lib
            cd /tmp
            wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
            tar -xzf ta-lib-0.4.0-src.tar.gz
            cd ta-lib/
            ./configure --prefix=/usr --quiet
            make --quiet
            sudo make install --quiet
            cd -
            rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

            # Установка Python wrapper
            source venv/bin/activate
            pip3 install ta-lib --quiet

            print_success "TA-Lib установлен"
        else
            print_info "Пропускаем установку TA-Lib"
            print_info "Установите вручную: https://github.com/mrjbq7/ta-lib"
        fi
    else
        print_info "Автоматическая установка TA-Lib доступна только на Linux"
        print_info "Для macOS: brew install ta-lib && pip3 install ta-lib"
    fi
}

# Запуск тестов
run_tests() {
    print_header "Запуск тестов"

    read -p "Запустить быстрый тест бота? [Y/n]: " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        source venv/bin/activate

        if [ -f "tests/quick_bot_test.py" ]; then
            print_info "Запуск быстрого теста (займёт ~30 секунд)..."
            python3 tests/quick_bot_test.py
        else
            print_warning "Тестовый файл не найден, пропускаем тесты"
        fi
    else
        print_info "Тесты пропущены"
    fi
}

# Финальные инструкции
print_final_instructions() {
    print_header "Установка завершена! 🎉"

    echo ""
    echo -e "${GREEN}Что дальше?${NC}"
    echo ""
    echo "1️⃣  Настройте .env файл:"
    echo "    nano .env"
    echo ""
    echo "2️⃣  Добавьте ваши Binance API ключи:"
    echo "    - BINANCE_API_KEY"
    echo "    - BINANCE_SECRET_KEY"
    echo ""
    echo "3️⃣  (Опционально) Настройте Telegram уведомления:"
    echo "    - TELEGRAM_BOT_TOKEN"
    echo "    - TELEGRAM_CHAT_ID"
    echo ""
    echo "4️⃣  Запустите бота в paper trading режиме:"
    echo "    source venv/bin/activate"
    echo "    python3 binance_bot_main.py --paper"
    echo ""
    echo "5️⃣  В другом терминале следите за логами:"
    echo "    tail -f logs/binance_bot.log"
    echo ""
    echo -e "${YELLOW}⚠️  ВАЖНО:${NC}"
    echo "    - Начните с PAPER TRADING режима"
    echo "    - Тестируйте минимум 1-2 недели"
    echo "    - Не используйте live trading без опыта!"
    echo ""
    echo -e "${BLUE}📚 Документация:${NC}"
    echo "    - Быстрый старт: cat QUICK_START.md"
    echo "    - Полное руководство: cat INSTALLATION_GUIDE.md"
    echo ""
    echo -e "${GREEN}Удачной торговли! 🚀📈${NC}"
    echo ""
}

# Основная функция
main() {
    clear

    print_header "🤖 BINANCE TRADING BOT - АВТОМАТИЧЕСКАЯ УСТАНОВКА"

    echo ""
    print_info "Этот скрипт выполнит:"
    echo "  1. Проверку системных требований"
    echo "  2. Очистку старого окружения"
    echo "  3. Создание нового виртуального окружения"
    echo "  4. Установку всех зависимостей"
    echo "  5. Настройку конфигурации"
    echo "  6. Запуск тестов"
    echo ""

    read -p "Продолжить? [Y/n]: " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Установка отменена"
        exit 0
    fi

    echo ""

    # Выполнение установки
    check_python
    clean_old_environment
    create_virtualenv
    install_dependencies
    create_directories
    check_permissions
    setup_env_file
    verify_installation
    install_talib
    run_tests
    print_final_instructions
}

# Обработка ошибок
trap 'print_error "Установка прервана!"; exit 1' INT TERM

# Запуск
main "$@"
