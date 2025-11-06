@echo off
REM ================================================================================
REM Скрипт просмотра логов в реальном времени для Windows
REM Версия: 1.0
REM ================================================================================

setlocal enabledelayedexpansion

REM Цвета
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

set "LOG_FILE=logs\binance_bot.log"

echo.
echo %BLUE%================================%NC%
echo %BLUE%   ПРОСМОТР ЛОГОВ БОТА%NC%
echo %BLUE%================================%NC%
echo.

REM Проверка существования лог файла
if not exist "%LOG_FILE%" (
    echo %RED%[ERROR]%NC% Лог файл не найден: %LOG_FILE%
    echo.
    echo Возможные причины:
    echo   1. Бот ещё не запускался
    echo   2. Бот запущен, но лог файл не создан
    echo.
    pause
    exit /b 1
)

echo %GREEN%[OK]%NC% Лог файл найден: %LOG_FILE%
echo %YELLOW%[INFO]%NC% Нажмите Ctrl+C для выхода
echo.
echo %BLUE%--------------------------------%NC%
echo.

REM Используем PowerShell для tail -f эквивалента
powershell -Command "Get-Content '%LOG_FILE%' -Wait -Tail 20 | ForEach-Object { if ($_ -match 'ERROR|CRITICAL') { Write-Host $_ -ForegroundColor Red } elseif ($_ -match 'WARNING') { Write-Host $_ -ForegroundColor Yellow } elseif ($_ -match 'SUCCESS|OK') { Write-Host $_ -ForegroundColor Green } else { Write-Host $_ } }"
