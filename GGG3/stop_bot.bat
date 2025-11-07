@echo off
REM ================================================================================
REM Скрипт остановки Binance Trading Bot для Windows
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
set "NC=[0m"

REM Файлы
set "PID_FILE=.bot.pid"

echo.
echo %BLUE%================================%NC%
echo %BLUE% ОСТАНОВКА BINANCE TRADING BOT%NC%
echo %BLUE%================================%NC%
echo.

REM ============================================================================
REM Проверка, что бот запущен
REM ============================================================================

if not exist "%PID_FILE%" (
    echo %RED%[ERROR]%NC% Бот не запущен ^(PID файл не найден^)
    echo.
    pause
    exit /b 1
)

set /p PID=<"%PID_FILE%"

REM Проверка существования процесса
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Процесс с PID %PID% не найден
    del "%PID_FILE%" 2>nul
    echo %YELLOW%[INFO]%NC% PID файл удалён
    echo.
    pause
    exit /b 1
)

REM ============================================================================
REM Остановка бота
REM ============================================================================

echo %YELLOW%[INFO]%NC% Остановка бота ^(PID: %PID%^)...

REM Попытка graceful shutdown через taskkill
taskkill /PID %PID% /T >nul 2>&1

REM Ждём до 10 секунд
set /a WAIT_COUNT=0
:wait_loop
timeout /t 1 /nobreak >nul
set /a WAIT_COUNT+=1

REM Проверяем, остановился ли процесс
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo %GREEN%[OK]%NC% Бот остановлен успешно
    del "%PID_FILE%" 2>nul
    echo.
    pause
    exit /b 0
)

if %WAIT_COUNT% LSS 10 goto wait_loop

REM ============================================================================
REM Принудительная остановка
REM ============================================================================

echo %YELLOW%[INFO]%NC% Процесс не остановился, используем принудительную остановку...

taskkill /F /PID %PID% /T >nul 2>&1

timeout /t 1 /nobreak >nul

REM Проверяем снова
tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
if errorlevel 1 (
    echo %GREEN%[OK]%NC% Бот остановлен принудительно
    del "%PID_FILE%" 2>nul
    echo.
    pause
    exit /b 0
) else (
    echo %RED%[ERROR]%NC% Не удалось остановить бота!
    echo.
    echo Попробуйте вручную через Диспетчер задач:
    echo   1. Откройте Диспетчер задач ^(Ctrl+Shift+Esc^)
    echo   2. Найдите процесс python.exe с PID %PID%
    echo   3. Завершите процесс
    echo.
    pause
    exit /b 1
)
