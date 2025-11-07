@echo off
REM ================================================================================
REM Автоматическая установка Binance Trading Bot для Windows
REM Версия: 1.0
REM ================================================================================

setlocal enabledelayedexpansion

REM Переход в директорию скрипта
cd /d "%~dp0"

REM Цвета для Windows
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo.
echo %BLUE%=================================================================================%NC%
echo %BLUE%         BINANCE TRADING BOT - АВТОМАТИЧЕСКАЯ УСТАНОВКА (WINDOWS)%NC%
echo %BLUE%=================================================================================%NC%
echo.

REM Проверка прав администратора (опционально)
echo %BLUE%[INFO]%NC% Проверка системных требований...
echo.

REM ============================================================================
REM Проверка Python
REM ============================================================================

echo %BLUE%[1/8]%NC% Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Python не найден!
    echo.
    echo Установите Python 3.9 или выше:
    echo   1. Скачайте: https://www.python.org/downloads/
    echo   2. При установке отметьте "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo %GREEN%[OK]%NC% Python %PYTHON_VERSION% найден
echo.

REM ============================================================================
REM Очистка старого окружения
REM ============================================================================

echo %BLUE%[2/8]%NC% Очистка старого окружения...

REM Деактивация текущего окружения (если активно)
if defined VIRTUAL_ENV (
    echo %YELLOW%[INFO]%NC% Деактивация текущего виртуального окружения...
    call deactivate 2>nul
)

REM Удаление старых виртуальных окружений
for %%d in (venv env .venv) do (
    if exist "%%d\" (
        echo %YELLOW%[INFO]%NC% Удаление %%d\...
        rmdir /s /q "%%d"
    )
)

REM Очистка кэша Python
echo %YELLOW%[INFO]%NC% Очистка кэша Python...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

echo %GREEN%[OK]%NC% Старое окружение очищено
echo.

REM ============================================================================
REM Создание виртуального окружения
REM ============================================================================

echo %BLUE%[3/8]%NC% Создание виртуального окружения...

python -m venv venv
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Не удалось создать виртуальное окружение!
    pause
    exit /b 1
)

echo %GREEN%[OK]%NC% Виртуальное окружение создано
echo.

REM ============================================================================
REM Активация виртуального окружения
REM ============================================================================

echo %BLUE%[4/8]%NC% Активация виртуального окружения...

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Не удалось активировать виртуальное окружение!
    pause
    exit /b 1
)

echo %GREEN%[OK]%NC% Виртуальное окружение активировано
echo.

REM ============================================================================
REM Обновление pip
REM ============================================================================

echo %BLUE%[5/8]%NC% Обновление pip, setuptools, wheel...

python -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 (
    echo %YELLOW%[WARNING]%NC% Не удалось обновить pip (продолжаем)
) else (
    echo %GREEN%[OK]%NC% pip обновлён
)
echo.

REM ============================================================================
REM Установка зависимостей
REM ============================================================================

echo %BLUE%[6/8]%NC% Установка зависимостей...
echo %YELLOW%[INFO]%NC% Это может занять 5-10 минут, пожалуйста подождите...
echo.

REM Установка основных зависимостей
echo %YELLOW%[INFO]%NC% Установка основных пакетов...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo %RED%[ERROR]%NC% Не удалось установить зависимости!
    echo.
    echo Проверьте:
    echo   1. Интернет соединение
    echo   2. Файл requirements.txt существует
    echo   3. Python версия >= 3.9
    echo.
    echo %GREEN%[INFO]%NC% Все индикаторы работают на чистом Python!
    echo   Не требуется ta-lib или другие C библиотеки.
    echo.
    pause
    exit /b 1
)

echo %GREEN%[OK]%NC% Основные зависимости установлены
echo.

REM ============================================================================
REM Создание директорий
REM ============================================================================

echo %BLUE%[7/8]%NC% Создание необходимых директорий...

if not exist "data" mkdir data
if not exist "data\positions" mkdir data\positions
if not exist "data\market_data" mkdir data\market_data
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "models\saved" mkdir models\saved

REM Создание .gitkeep
type nul > models\saved\.gitkeep

echo %GREEN%[OK]%NC% Директории созданы
echo.

REM ============================================================================
REM Настройка .env файла
REM ============================================================================

echo %BLUE%[8/8]%NC% Настройка конфигурации...

if not exist ".env" (
    if exist ".env.example" (
        echo %YELLOW%[INFO]%NC% Создание .env из .env.example...
        copy .env.example .env >nul
        echo %GREEN%[OK]%NC% .env файл создан
        echo.
        echo %YELLOW%[ВАЖНО]%NC% Не забудьте отредактировать .env файл и добавить ваши API ключи!
        echo.
        echo Откройте файл: notepad .env
    ) else (
        echo %RED%[ERROR]%NC% .env.example не найден!
    )
) else (
    echo %YELLOW%[INFO]%NC% .env файл уже существует, пропускаем...
)
echo.

REM ============================================================================
REM Проверка установки
REM ============================================================================

echo %BLUE%=================================================================================%NC%
echo %BLUE%                           ПРОВЕРКА УСТАНОВКИ%NC%
echo %BLUE%=================================================================================%NC%
echo.

if exist "verify_dependencies.py" (
    echo %YELLOW%[INFO]%NC% Запуск проверки зависимостей...
    echo.
    python verify_dependencies.py
    echo.
) else (
    echo %YELLOW%[WARNING]%NC% verify_dependencies.py не найден, пропускаем проверку
)

REM ============================================================================
REM Запуск тестов (опционально)
REM ============================================================================

echo.
set /p RUN_TESTS="Запустить быстрый тест бота? (Y/n): "
if /i "%RUN_TESTS%"=="n" goto skip_tests

if exist "tests\quick_bot_test.py" (
    echo.
    echo %YELLOW%[INFO]%NC% Запуск быстрого теста (займёт ~30 секунд)...
    echo.
    python tests\quick_bot_test.py
) else (
    echo %YELLOW%[WARNING]%NC% Тестовый файл не найден, пропускаем тесты
)

:skip_tests

REM ============================================================================
REM Финальные инструкции
REM ============================================================================

echo.
echo %BLUE%=================================================================================%NC%
echo %GREEN%                    УСТАНОВКА ЗАВЕРШЕНА! 🎉%NC%
echo %BLUE%=================================================================================%NC%
echo.
echo %GREEN%Что дальше?%NC%
echo.
echo %GREEN%[ГОТОВО К ЗАПУСКУ!]%NC% Для paper trading API ключи НЕ нужны!
echo.
echo 1. %YELLOW%[ОПЦИОНАЛЬНО]%NC% Проверьте .env файл:
echo    notepad .env
echo.
echo    %GREEN%Для paper trading оставьте как есть:%NC%
echo    - PAPER_TRADING_MODE=true
echo    - USE_TESTNET=false (для реальных данных рынка!)
echo    - BINANCE_API_KEY= (пустое)
echo    - BINANCE_SECRET_KEY= (пустое)
echo.
echo 2. Запустите бота в paper trading режиме:
echo    start_bot.bat
echo.
echo 3. В другом окне следите за логами:
echo    tail_logs.bat
echo.
echo %BLUE%[i]%NC% API ключи нужны ТОЛЬКО для live trading (реальные деньги)
echo.
echo %YELLOW%ВАЖНО:%NC%
echo    - Начните с PAPER TRADING режима
echo    - Тестируйте минимум 1-2 недели
echo    - Не используйте live trading без опыта!
echo.
echo %BLUE%Документация:%NC%
echo    - Быстрый старт: type QUICK_START_WINDOWS.md
echo    - Полное руководство: type INSTALLATION_GUIDE_WINDOWS.md
echo.
echo %GREEN%Удачной торговли! 🚀📈%NC%
echo.
echo %YELLOW%Нажмите любую клавишу для выхода...%NC%
pause >nul
