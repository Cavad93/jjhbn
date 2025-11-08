@echo off
REM Отключаем буферизацию вывода Python для Windows
set PYTHONUNBUFFERED=1

REM Запускаем бота с флагом -u (unbuffered output)
python -u binance_bot_main.py --paper
