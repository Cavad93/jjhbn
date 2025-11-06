@echo off
REM ============================================================================
REM 🧹 Скрипт очистки старого бота для Windows Server 2012 R2
REM ============================================================================
REM Автор: Trading Bot Team
REM Версия: 1.0
REM Дата: 2025-11-06
REM ============================================================================

setlocal enabledelayedexpansion

REM Цвета для вывода (ANSI escape codes)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "MAGENTA=[95m"
set "CYAN=[96m"
set "NC=[0m"

REM ============================================================================
REM Заголовок
REM ============================================================================
cls
echo.
echo %CYAN%============================================================================%NC%
echo %CYAN%  🧹 Удаление старого бота PostgreSQL 12%NC%
echo %CYAN%============================================================================%NC%
echo.
echo %YELLOW%[!] Этот скрипт:%NC%
echo    1. Остановит все процессы Python
echo    2. Удалит старые виртуальные окружения
echo    3. Очистит кэш pip
echo    4. Подготовит систему для установки нового бота
echo.
echo %RED%[!] ВНИМАНИЕ: Убедитесь, что сделали резервную копию важных данных!%NC%
echo.

REM ============================================================================
REM Запрос подтверждения
REM ============================================================================
set /p confirm="Продолжить? (y/n): "
if /i not "%confirm%"=="y" (
    echo.
    echo %RED%[ОТМЕНА]%NC% Операция отменена пользователем
    pause
    exit /b 1
)

echo.
echo %GREEN%[НАЧАЛО]%NC% Запуск процесса очистки...
echo.

REM ============================================================================
REM ШАГ 1: Остановить все процессы Python
REM ============================================================================
echo %CYAN%[ШАГ 1/6]%NC% Остановка процессов Python...

tasklist | findstr /i python >nul 2>&1
if %errorlevel% equ 0 (
    echo %YELLOW%[i]%NC% Найдены активные процессы Python, останавливаем...
    taskkill /F /IM python.exe >nul 2>&1
    taskkill /F /IM pythonw.exe >nul 2>&1
    timeout /t 2 /nobreak >nul
    echo %GREEN%[✓]%NC% Процессы Python остановлены
) else (
    echo %GREEN%[✓]%NC% Процессы Python не найдены
)

echo.

REM ============================================================================
REM ШАГ 2: Запросить путь к старому боту
REM ============================================================================
echo %CYAN%[ШАГ 2/6]%NC% Определение пути к старому боту...
echo.
echo %YELLOW%[?]%NC% Введите полный путь к папке СТАРОГО бота
echo %YELLOW%[?]%NC% Например: C:\Users\Admin\Desktop\old_bot
echo %YELLOW%[?]%NC% Или нажмите Enter, чтобы пропустить этот шаг
echo.

set /p old_bot_path="Путь к старому боту: "

if "%old_bot_path%"=="" (
    echo %YELLOW%[i]%NC% Пропуск удаления старого окружения
    goto skip_old_env
)

if not exist "%old_bot_path%" (
    echo %RED%[✗]%NC% Папка не найдена: %old_bot_path%
    echo %YELLOW%[i]%NC% Пропуск удаления старого окружения
    goto skip_old_env
)

echo %GREEN%[✓]%NC% Найдена папка: %old_bot_path%
echo.

REM ============================================================================
REM ШАГ 3: Удалить старые виртуальные окружения
REM ============================================================================
echo %CYAN%[ШАГ 3/6]%NC% Удаление старых виртуальных окружений...

cd /d "%old_bot_path%" || (
    echo %RED%[✗]%NC% Не удалось перейти в папку
    goto skip_old_env
)

set "found_venv=0"

if exist "venv" (
    echo %YELLOW%[i]%NC% Удаление папки: venv
    rmdir /S /Q venv 2>nul
    if !errorlevel! equ 0 (
        echo %GREEN%[✓]%NC% Удалено: venv
        set "found_venv=1"
    ) else (
        echo %RED%[✗]%NC% Ошибка при удалении venv
    )
)

if exist ".venv" (
    echo %YELLOW%[i]%NC% Удаление папки: .venv
    rmdir /S /Q .venv 2>nul
    if !errorlevel! equ 0 (
        echo %GREEN%[✓]%NC% Удалено: .venv
        set "found_venv=1"
    ) else (
        echo %RED%[✗]%NC% Ошибка при удалении .venv
    )
)

if exist "env" (
    echo %YELLOW%[i]%NC% Удаление папки: env
    rmdir /S /Q env 2>nul
    if !errorlevel! equ 0 (
        echo %GREEN%[✓]%NC% Удалено: env
        set "found_venv=1"
    ) else (
        echo %RED%[✗]%NC% Ошибка при удалении env
    )
)

if %found_venv% equ 0 (
    echo %YELLOW%[i]%NC% Виртуальные окружения не найдены
)

echo.

:skip_old_env

REM ============================================================================
REM ШАГ 4: Очистить кэш pip
REM ============================================================================
echo %CYAN%[ШАГ 4/6]%NC% Очистка кэша pip...

python -m pip cache purge >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%[✓]%NC% Кэш pip очищен
) else (
    echo %YELLOW%[!]%NC% Не удалось очистить кэш pip (возможно, не установлен pip)
)

echo.

REM ============================================================================
REM ШАГ 5: Удалить глобальные зависимости PostgreSQL (опционально)
REM ============================================================================
echo %CYAN%[ШАГ 5/6]%NC% Удаление глобальных зависимостей PostgreSQL...
echo.
echo %RED%[!] ВНИМАНИЕ:%NC% Это удалит PostgreSQL библиотеки для ВСЕХ Python проектов!
echo %YELLOW%[i]%NC% Рекомендуется пропустить этот шаг, если используете виртуальные окружения
echo.

set /p remove_global="Удалить глобальные зависимости PostgreSQL? (y/n): "
if /i "%remove_global%"=="y" (
    echo.
    echo %YELLOW%[i]%NC% Удаление psycopg2-binary...
    python -m pip uninstall -y psycopg2-binary >nul 2>&1

    echo %YELLOW%[i]%NC% Удаление psycopg2...
    python -m pip uninstall -y psycopg2 >nul 2>&1

    echo %YELLOW%[i]%NC% Удаление asyncpg...
    python -m pip uninstall -y asyncpg >nul 2>&1

    echo %GREEN%[✓]%NC% Глобальные зависимости PostgreSQL удалены
) else (
    echo %YELLOW%[i]%NC% Пропуск удаления глобальных зависимостей
)

echo.

REM ============================================================================
REM ШАГ 6: Информация о следующих шагах
REM ============================================================================
echo %CYAN%[ШАГ 6/6]%NC% Подготовка к установке нового бота...
echo.

REM Определить текущую папку
set "current_dir=%cd%"

echo %GREEN%[✓]%NC% Очистка завершена успешно!
echo.
echo %CYAN%============================================================================%NC%
echo %GREEN%  Следующие шаги:%NC%
echo %CYAN%============================================================================%NC%
echo.
echo %YELLOW%1.%NC% Перейдите в папку нового бота:
echo    %CYAN%cd C:\path\to\jjhbn\GGG3%NC%
echo.
echo %YELLOW%2.%NC% Запустите автоматическую установку:
echo    %CYAN%install.bat%NC%
echo.
echo %YELLOW%3.%NC% Настройте конфигурацию (опционально):
echo    %CYAN%notepad .env%NC%
echo.
echo %YELLOW%4.%NC% Запустите нового бота:
echo    %CYAN%start_bot.bat%NC%
echo.
echo %CYAN%============================================================================%NC%
echo.

REM ============================================================================
REM Предложить автоматический переход
REM ============================================================================
echo %YELLOW%[?]%NC% Хотите сразу перейти к установке нового бота?
echo.

set /p goto_install="Перейти к установке? (y/n): "
if /i "%goto_install%"=="y" (
    echo.
    echo %GREEN%[ПЕРЕХОД]%NC% Запрос пути к новому боту...
    echo.
    set /p new_bot_path="Введите путь к новому боту (jjhbn\GGG3): "

    if "!new_bot_path!"=="" (
        echo %RED%[✗]%NC% Путь не указан
        goto end_script
    )

    if not exist "!new_bot_path!" (
        echo %RED%[✗]%NC% Папка не найдена: !new_bot_path!
        goto end_script
    )

    cd /d "!new_bot_path!" || (
        echo %RED%[✗]%NC% Не удалось перейти в папку
        goto end_script
    )

    echo %GREEN%[✓]%NC% Переход в папку: !new_bot_path!
    echo.

    REM Проверить наличие install.bat
    if exist "install.bat" (
        echo %GREEN%[ЗАПУСК]%NC% Запуск install.bat...
        echo.
        call install.bat
    ) else (
        echo %RED%[✗]%NC% Файл install.bat не найден
        echo %YELLOW%[i]%NC% Убедитесь, что вы в правильной папке
    )
)

:end_script
echo.
echo %GREEN%[ГОТОВО]%NC% Скрипт очистки завершён
echo.
pause
exit /b 0
