@echo off
REM ================================================================================
REM Скрипт проверки статуса Binance Trading Bot для Windows
REM Версия: 1.0
REM ================================================================================

setlocal enabledelayedexpansion

REM Переход в директорию скрипта
cd /d "%~dp0"

REM Цвета
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "CYAN=[96m"
set "NC=[0m"

REM Файлы
set "PID_FILE=.bot.pid"
set "LOG_FILE=logs\binance_bot.log"
set "POSITIONS_FILE=data\positions\positions.json"

cls
echo.
echo %BLUE%╔════════════════════════════════════════╗%NC%
echo %BLUE%║   BINANCE TRADING BOT - СТАТУС         ║%NC%
echo %BLUE%╚════════════════════════════════════════╝%NC%
echo.

REM ============================================================================
REM Проверка статуса процесса
REM ============================================================================

echo %BLUE%================================%NC%
echo %BLUE%       Статус процесса%NC%
echo %BLUE%================================%NC%
echo.

if not exist "%PID_FILE%" (
    echo %RED%[❌] Бот НЕ ЗАПУЩЕН%NC% ^(PID файл не найден^)
    set BOT_RUNNING=0
) else (
    set /p PID=<"%PID_FILE%"

    REM Проверка существования процесса
    tasklist /FI "PID eq !PID!" 2>nul | find "!PID!" >nul
    if errorlevel 1 (
        echo %RED%[❌] Бот НЕ РАБОТАЕТ%NC% ^(процесс PID !PID! не найден^)
        echo.
        echo %CYAN%[INFO]%NC% PID файл существует, но процесс не запущен
        echo %CYAN%[INFO]%NC% Возможно бот был остановлен некорректно
        echo %CYAN%[INFO]%NC% Очистите PID файл: del .bot.pid
        set BOT_RUNNING=0
    ) else (
        echo %GREEN%[✅] Бот РАБОТАЕТ%NC% ^(PID: !PID!^)
        set BOT_RUNNING=1

        echo.
        echo %CYAN%Информация о процессе:%NC%

        REM CPU и память
        for /f "tokens=2,5" %%a in ('tasklist /FI "PID eq !PID!" /FO TABLE /NH') do (
            echo   PID:        %%a
            echo   Память:     %%b
        )

        REM Uptime (примерное, через WMIC)
        for /f "skip=1 tokens=1-6" %%a in ('wmic process where "processid=!PID!" get CreationDate /value ^| findstr /R "="') do (
            set CREATION_DATE=%%a
        )
        if defined CREATION_DATE (
            for /f "tokens=2 delims==" %%a in ("!CREATION_DATE!") do set START_TIME=%%a
            echo   Запущен:    !START_TIME:~0,4!-!START_TIME:~4,2!-!START_TIME:~6,2! !START_TIME:~8,2!:!START_TIME:~10,2!:!START_TIME:~12,2!
        )
    )
)

REM ============================================================================
REM Проверка конфигурации
REM ============================================================================

echo.
echo %BLUE%================================%NC%
echo %BLUE%        Конфигурация%NC%
echo %BLUE%================================%NC%
echo.

if exist ".env" (
    echo %GREEN%[✅] .env файл найден%NC%
    echo.
    echo %CYAN%Настроенные переменные:%NC%

    REM Проверка ключевых переменных
    findstr /B /C:"BINANCE_API_KEY=" .env >nul 2>&1
    if errorlevel 1 (
        echo   %RED%[✗]%NC% BINANCE_API_KEY ^(не настроен^)
    ) else (
        findstr /B /C:"BINANCE_API_KEY=your" .env >nul 2>&1
        if errorlevel 1 (
            echo   %GREEN%[✓]%NC% BINANCE_API_KEY
        ) else (
            echo   %YELLOW%[✗]%NC% BINANCE_API_KEY ^(не настроен^)
        )
    )

    findstr /B /C:"BINANCE_SECRET_KEY=" .env >nul 2>&1
    if errorlevel 1 (
        echo   %RED%[✗]%NC% BINANCE_SECRET_KEY ^(не настроен^)
    ) else (
        findstr /B /C:"BINANCE_SECRET_KEY=your" .env >nul 2>&1
        if errorlevel 1 (
            echo   %GREEN%[✓]%NC% BINANCE_SECRET_KEY
        ) else (
            echo   %YELLOW%[✗]%NC% BINANCE_SECRET_KEY ^(не настроен^)
        )
    )

    findstr /B /C:"TELEGRAM_BOT_TOKEN=" .env >nul 2>&1
    if errorlevel 1 (
        echo   %YELLOW%[✗]%NC% TELEGRAM_BOT_TOKEN ^(не настроен^)
    ) else (
        echo   %GREEN%[✓]%NC% TELEGRAM_BOT_TOKEN
    )

    REM Режим
    for /f "tokens=2 delims==" %%a in ('findstr /B /C:"PAPER_TRADING_MODE=" .env 2^>nul') do set MODE=%%a
    if "!MODE!"=="true" (
        echo   %GREEN%[✓]%NC% Режим: PAPER TRADING
    ) else if "!MODE!"=="false" (
        echo   %YELLOW%[⚠]%NC% Режим: LIVE TRADING
    ) else (
        echo   %CYAN%[?]%NC% Режим: не определён ^(по умолчанию PAPER^)
    )
) else (
    echo %RED%[❌] .env файл не найден!%NC%
)

REM ============================================================================
REM Проверка использования ресурсов (только если бот работает)
REM ============================================================================

if "%BOT_RUNNING%"=="1" (
    echo.
    echo %BLUE%================================%NC%
    echo %BLUE%   Использование ресурсов%NC%
    echo %BLUE%================================%NC%
    echo.

    echo %CYAN%Система:%NC%
    echo.

    REM Память
    for /f "skip=1 tokens=2 delims=," %%a in ('wmic os get FreePhysicalMemory^,TotalVisibleMemorySize /format:csv') do (
        set /a TOTAL_MEM=%%a/1024
        set /a FREE_MEM=%%b/1024
        set /a USED_MEM=!TOTAL_MEM!-!FREE_MEM!
        set /a MEM_PERCENT=!USED_MEM!*100/!TOTAL_MEM!
        echo   RAM: !USED_MEM! MB / !TOTAL_MEM! MB используется ^(!MEM_PERCENT!%%^)
    )

    REM CPU (общая нагрузка)
    for /f "skip=1 tokens=2 delims=," %%a in ('wmic cpu get LoadPercentage /format:csv') do (
        echo   CPU: %%a%%
    )

    echo.
    echo %CYAN%Бот:%NC%
    echo.

    REM Память бота
    for /f "skip=1 tokens=5" %%a in ('tasklist /FI "PID eq !PID!" /FO TABLE /NH') do (
        set MEM_KB=%%a
        set MEM_KB=!MEM_KB:,=!
        set /a MEM_MB=!MEM_KB!/1024
        echo   Память: !MEM_MB! MB
    )
)

REM ============================================================================
REM Проверка открытых позиций
REM ============================================================================

echo.
echo %BLUE%================================%NC%
echo %BLUE%      Открытые позиции%NC%
echo %BLUE%================================%NC%
echo.

if not exist "%POSITIONS_FILE%" (
    echo %YELLOW%[⚠]%NC% Файл позиций не найден: %POSITIONS_FILE%
) else (
    REM Простой вывод через Python (если доступен)
    python -c "import json; f=open('%POSITIONS_FILE%'); d=json.load(f); print(f\"Открыто позиций:  {len(d.get('open_positions', []))}\"); print(f\"Закрыто (последние): {len(d.get('closed_positions', []))}\"); print(f\"Закрыто всего:    {d.get('metadata', {}).get('total_closed', 0)}\")" 2>nul
    if errorlevel 1 (
        REM Fallback - показываем размер файла
        echo %CYAN%[INFO]%NC% Файл позиций существует (для просмотра используйте Python^)
        dir "%POSITIONS_FILE%" | findstr /C:"positions.json"
    )
)

REM ============================================================================
REM Проверка последних логов (только если бот работает)
REM ============================================================================

if "%BOT_RUNNING%"=="1" (
    echo.
    echo %BLUE%================================%NC%
    echo %BLUE%       Последние логи%NC%
    echo %BLUE%================================%NC%
    echo.

    if not exist "%LOG_FILE%" (
        echo %YELLOW%[⚠]%NC% Лог файл не найден: %LOG_FILE%
    ) else (
        echo %CYAN%Последние 10 строк из лога:%NC%
        echo.

        REM Показываем последние 10 строк (через PowerShell)
        powershell -Command "Get-Content '%LOG_FILE%' -Tail 10"

        echo.
        echo %CYAN%[INFO]%NC% Для просмотра всех логов: tail_logs.bat
    )
)

REM ============================================================================
REM Быстрые команды
REM ============================================================================

echo.
echo %BLUE%================================%NC%
echo %BLUE%      Быстрые команды%NC%
echo %BLUE%================================%NC%
echo.

if "%BOT_RUNNING%"=="1" (
    echo %CYAN%Остановить бота:%NC%     stop_bot.bat
    echo %CYAN%Просмотр логов:%NC%      tail_logs.bat
) else (
    echo %CYAN%Запустить бота:%NC%      start_bot.bat
    echo %CYAN%Запуск в фоне:%NC%       start_bot.bat --background
)

echo %CYAN%Обновить статус:%NC%     status_bot.bat

echo.
pause
