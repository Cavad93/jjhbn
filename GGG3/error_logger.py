# -*- coding: utf-8 -*-
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

_logger = None

def setup_error_logging(log_dir=".", filename="errors.log",
                        max_bytes=10_000_000, backup_count=5) -> logging.Logger:
    """
    Инициализация ротируемого error-логгера. Пишет только уровни ERROR+.
    Также перехватывает все необработанные исключения (main и потоки).
    """
    global _logger
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)

    logger = logging.getLogger("errors")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers[:] = []

    fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count,
                             encoding="utf-8", delay=True)
    fh.setLevel(logging.ERROR)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(process)d | %(threadName)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Глобальный перехват неотловленных исключений
    def _excepthook(exc_type, exc, tb):
        try:
            logger.error("Uncaught exception", exc_info=(exc_type, exc, tb))
        finally:
            sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _excepthook

    # Перехват исключений в потоках (Py3.8+)
    def _thread_excepthook(args):
        try:
            logger.error("Uncaught thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        finally:
            if hasattr(threading, "__excepthook__"):
                threading.__excepthook__(args)
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook

    logging.captureWarnings(True)
    _logger = logger
    return logger

def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        return setup_error_logging()
    return _logger

def log_exception(msg: str = "Unhandled exception"):
    """Логирует текущий стек как ERROR (используйте внутри except)."""
    get_logger().exception(msg)
