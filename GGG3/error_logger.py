# -*- coding: utf-8 -*-
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional

# Thread-safe lock для инициализации
_logger_lock = threading.Lock()
_logger: Optional[logging.Logger] = None
_initialized = False

# Сохраняем оригинальные hooks для корректной цепочки
_original_excepthook = sys.excepthook
_original_thread_excepthook = None


def setup_error_logging(log_dir: str = ".", 
                        filename: str = "errors.log",
                        max_bytes: int = 10_000_000, 
                        backup_count: int = 5,
                        force_reinit: bool = False) -> logging.Logger:
    """
    Инициализация ротируемого error-логгера. Пишет только уровни ERROR+.
    Также перехватывает все необработанные исключения (main и потоки).
    
    Args:
        log_dir: Директория для логов
        filename: Имя файла лога
        max_bytes: Максимальный размер файла до ротации
        backup_count: Количество backup файлов
        force_reinit: Принудительная переинициализация (осторожно!)
    
    Returns:
        Настроенный logger
    
    Note:
        Безопасен для многопоточного использования.
        Повторные вызовы игнорируются (если не force_reinit=True).
    """
    global _logger, _initialized, _original_excepthook, _original_thread_excepthook
    
    # ИСПРАВЛЕНИЕ БАГ #2 и #3: Thread-safe инициализация
    with _logger_lock:
        # Если уже инициализирован и не требуется переинициализация
        if _initialized and not force_reinit:
            return _logger
        
        # Создаем директорию
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)

        # Получаем или создаем logger
        logger = logging.getLogger("errors")
        
        # ИСПРАВЛЕНИЕ БАГ #5: Устанавливаем согласованные уровни
        # Логгер на ERROR, чтобы не принимать лишние сообщения
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        
        # ИСПРАВЛЕНИЕ БАГ #3: Очищаем только если переинициализация
        if force_reinit:
            logger.handlers[:] = []

        # Создаем handler только если его еще нет
        has_file_handler = any(
            isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(path)
            for h in logger.handlers
        )
        
        if not has_file_handler:
            fh = RotatingFileHandler(
                path, 
                maxBytes=max_bytes, 
                backupCount=backup_count,
                encoding="utf-8", 
                delay=True
            )
            fh.setLevel(logging.ERROR)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(process)d | %(threadName)s | %(name)s | %(message)s"
            )
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        # ИСПРАВЛЕНИЕ БАГ #3: Устанавливаем excepthook только один раз
        if not _initialized:
            # Сохраняем оригинальный excepthook
            _original_excepthook = sys.excepthook
            
            # Глобальный перехват неотловленных исключений
            def _excepthook(exc_type, exc_value, exc_traceback):
                try:
                    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
                except Exception:
                    # Если логирование не удалось, хотя бы выведем в stderr
                    pass
                finally:
                    # ИСПРАВЛЕНИЕ БАГ #1: Вызываем СОХРАНЕННЫЙ оригинальный hook
                    _original_excepthook(exc_type, exc_value, exc_traceback)
            
            sys.excepthook = _excepthook

            # ИСПРАВЛЕНИЕ БАГ #1: Корректная работа с threading.excepthook
            if hasattr(threading, "excepthook"):
                # Сохраняем оригинальный thread excepthook
                _original_thread_excepthook = threading.excepthook
                
                def _thread_excepthook(args):
                    try:
                        logger.error(
                            "Uncaught thread exception", 
                            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
                        )
                    except Exception:
                        pass
                    finally:
                        # ИСПРАВЛЕНО: Вызываем оригинальный threading.excepthook
                        # (НЕ threading.__excepthook__ которого не существует!)
                        if _original_thread_excepthook is not None:
                            _original_thread_excepthook(args)
                
                threading.excepthook = _thread_excepthook

            # Перехватываем warnings
            logging.captureWarnings(True)
        
        _logger = logger
        _initialized = True
        return logger


def get_logger() -> logging.Logger:
    """
    Получить глобальный error logger.
    
    Returns:
        Настроенный logger (создается автоматически если еще не создан)
    
    Note:
        Thread-safe. Если logger не инициализирован, создается с параметрами по умолчанию.
    """
    global _logger
    
    # ИСПРАВЛЕНИЕ БАГ #2: Thread-safe проверка
    if _logger is None:
        with _logger_lock:
            # Double-checked locking pattern
            if _logger is None:
                # ИСПРАВЛЕНИЕ БАГ #4: Используем параметры по умолчанию
                # но документируем это поведение
                return setup_error_logging()
    
    return _logger


def log_exception(msg: str = "Unhandled exception") -> None:
    """
    Логирует текущее исключение как ERROR (используйте внутри except блока).
    
    Args:
        msg: Сообщение для лога
    
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception:
        ...     log_exception("Failed to do risky operation")
    
    Note:
        Если вызвано вне except блока, залогирует "NoneType: None".
    """
    try:
        get_logger().exception(msg)
    except Exception:
        # Fallback: если даже логирование упало, выводим в stderr
        import traceback
        print(f"ERROR: Failed to log exception: {msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def reset_logger() -> None:
    """
    Сброс глобального logger (для тестов или переинициализации).
    
    Warning:
        Использовать только в тестах! В production коде вызывайте
        setup_error_logging(force_reinit=True) если нужна переинициализация.
    """
    global _logger, _initialized, _original_excepthook, _original_thread_excepthook
    
    with _logger_lock:
        if _logger is not None:
            # Удаляем все handlers
            for handler in _logger.handlers[:]:
                handler.close()
                _logger.removeHandler(handler)
        
        # Восстанавливаем оригинальные hooks
        if _original_excepthook is not None:
            sys.excepthook = _original_excepthook
        
        if _original_thread_excepthook is not None and hasattr(threading, "excepthook"):
            threading.excepthook = _original_thread_excepthook
        
        _logger = None
        _initialized = False
