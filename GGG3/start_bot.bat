@echo off
REM ================================================================================
REM Скрипт запуска Binance Trading Bot для Windows
REM Версия: 1.0
REM ================================================================================

REM Установка UTF-8 кодировки для корректного отображения русских символов
chcp 65001 >nul 2>&1

setlocal enabledelayedexpansion

REM Цвета
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Файлы
set "PID_FILE=.bot.pid"
set "LOG_FILE=logs\binance_bot.log"

REM ============================================================================
REM Проверка, что бот не запущен
REM ============================================================================

if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"

    REM Проверяем существование процесса
    tasklist /FI "PID eq !OLD_PID!" 2>nul | find "!OLD_PID!" >nul
    if !errorlevel! equ 0 (
        echo %RED%[ERROR]%NC% Бот уже запущен ^(PID: !OLD_PID!^)
        echo.
        echo Для остановки используйте:
        echo   stop_bot.bat
        echo.
        echo Для проверки статуса:
        echo   status_bot.bat
        echo.
        pause
        exit /b 1
    ) else (
        REM PID файл существует, но процесс не запущен - очищаем
        del "%PID_FILE%" 2>nul
    )
)

REM ============================================================================
REM Проверка виртуального окружения
REM ============================================================================

if not exist "venv\" (
    echo %RED%[ERROR]%NC% Виртуальное окружение не найдено!
    echo Запустите установку: install.bat
    echo.
    pause
    exit /b 1
)

REM ============================================================================
REM Проверка .env файла
REM ============================================================================

if not exist ".env" (
    echo %RED%[ERROR]%NC% .env файл не найден!
    echo.
    echo Создайте .env файл:
    echo   copy .env.example .env
    echo   notepad .env
    echo.
    pause
    exit /b 1
)

REM ============================================================================
REM Создание директорий для логов
REM ============================================================================

if not exist "logs" mkdir logs
if not exist "data\positions" mkdir data\positions

REM ============================================================================
REM Определение режима
REM ============================================================================

set "MODE=paper"
set "MODE_ARG=--paper"

REM Проверка аргументов командной строки
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--live" (
    set "MODE=live"
    set "MODE_ARG=--live"
)
if /i "%~1"=="-live" (
    set "MODE=live"
    set "MODE_ARG=--live"
)
if /i "%~1"=="--background" set "BACKGROUND=1"
if /i "%~1"=="-b" set "BACKGROUND=1"
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
shift
goto parse_args

:args_done

REM ============================================================================
REM Запуск бота
REM ============================================================================

echo.
echo %BLUE%================================%NC%
echo %BLUE%   ЗАПУСК BINANCE TRADING BOT%NC%
echo %BLUE%================================%NC%
echo.

if "%MODE%"=="live" (
    echo %YELLOW%[WARNING]%NC% LIVE TRADING РЕЖИМ!
    echo %YELLOW%[WARNING]%NC% Будут использованы РЕАЛЬНЫЕ деньги!
    echo.
    set /p CONFIRM="Вы уверены? Введите 'YES' для продолжения: "
    if not "!CONFIRM!"=="YES" (
        echo %YELLOW%[INFO]%NC% Запуск отменён
        pause
        exit /b 0
    )
) else (
    echo %YELLOW%[INFO]%NC% PAPER TRADING режим
)

echo.
echo %YELLOW%[INFO]%NC% Активация виртуального окружения...
call venv\Scripts\activate.bat

echo %YELLOW%[INFO]%NC% Запуск бота...
echo.

if defined BACKGROUND (
    REM Фоновый режим
    echo %YELLOW%[INFO]%NC% Запуск в фоновом режиме...

    REM Запуск через start с перенаправлением в лог
    start /B "" python binance_bot_main.py %MODE_ARG% >> "%LOG_FILE%" 2>&1

    REM Небольшая задержка для получения PID
    timeout /t 2 /nobreak >nul

    REM Получение PID последнего процесса Python
    for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /I "PID:"') do set LAST_PID=%%a

    REM Сохранение PID
    echo !LAST_PID! > "%PID_FILE%"

    echo %GREEN%[OK]%NC% Бот запущен в фоновом режиме ^(PID: !LAST_PID!^)
    echo.
    echo Для просмотра логов:
    echo   tail_logs.bat
    echo.
    echo Для остановки:
    echo   stop_bot.bat
    echo.
    echo Для проверки статуса:
    echo   status_bot.bat
    echo.
    pause
) else (
    REM Интерактивный режим
    echo %YELLOW%[INFO]%NC% Интерактивный режим (нажмите Ctrl+C для остановки)
    echo.
    python binance_bot_main.py %MODE_ARG%
)

goto :eof

REM ============================================================================
REM Справка
REM ============================================================================

:show_help
echo Использование: start_bot.bat [OPTIONS]
echo.
echo Опции:
echo   --paper          Запуск в paper trading режиме ^(по умолчанию^)
echo   --live           Запуск в live trading режиме ^(ОСТОРОЖНО!^)
echo   --background, -b Запуск в фоновом режиме
echo   --help, -h       Показать эту справку
echo.
echo Примеры:
echo   start_bot.bat                    # Paper trading, интерактивный режим
echo   start_bot.bat --background       # Paper trading, фоновый режим
echo   start_bot.bat --live             # Live trading, интерактивный режим
echo   start_bot.bat --live --background # Live trading, фоновый режим
echo.
pause
exit /b 0
