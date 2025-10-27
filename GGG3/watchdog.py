# -*- coding: utf-8 -*-
import os
import sys
import time
import subprocess

from error_logger import setup_error_logging, get_logger, log_exception

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bot_script = os.path.join(base_dir, "bnbusdrt6.py")

    setup_error_logging(log_dir=base_dir, filename="errors.log")
    logger = get_logger()

    backoff = 5
    max_backoff = 300
    crash_count = 0

    while True:
        start_ts = time.time()
        env = os.environ.copy()
        env["RUN_UNDER_WATCHDOG"] = "1"

        try:
            proc = subprocess.Popen([sys.executable, bot_script], cwd=base_dir, env=env)
            rc = proc.wait()
        except KeyboardInterrupt:
            print("⏹️ Watchdog остановлен пользователем (Ctrl+C).")
            try:
                proc.terminate()
            except Exception:
                from error_logger import log_exception
                log_exception("Failed to terminate process")
            break
        except Exception:
            log_exception("Watchdog: ошибка при запуске дочернего процесса")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)
            crash_count += 1
            continue

        if rc == 0:
            print("✅ Bot завершился нормально (код 0). Watchdog завершается.")
            break

        uptime = time.time() - start_ts
        crash_count += 1
        logger.error(f"Bot crashed: exit={rc}, uptime={uptime:.1f}s, restart#{crash_count}")

        if uptime > 600:
            backoff = 5
        else:
            backoff = min(max_backoff, backoff * 2)

        print(f"↻ Перезапуск через {backoff} сек...")
        time.sleep(backoff)

if __name__ == "__main__":
    main()
