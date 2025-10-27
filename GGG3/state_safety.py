# state_safety.py
import os, json, hashlib, tempfile, numpy as np
from typing import Optional, Dict, Any

MAGIC = "MLSTATE_v1"

def atomic_write_bytes(path: str, data: bytes):
    """Атомарная запись байтов в файл с fsync."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        # Записываем все байты (os.write может записать меньше)
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)

def file_sha256(path: str) -> str:
    """Вычисляет SHA256 хеш файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_save_json(path: str, payload: dict):
    """
    Атомарное сохранение JSON с защитой от повреждения:
    - Добавляет _magic, _version, _sha256
    - Создает .bak резервную копию
    - Атомарная замена через os.replace
    """
    payload = dict(payload)
    if "_magic" not in payload:
        payload["_magic"] = MAGIC
    payload["_version"] = 1
    
    # Вычисляем SHA256 без поля _sha256
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload["_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    
    # Финальная сериализация с _sha256
    body_final = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    data = body_final.encode("utf-8")
    
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")  # binary mode
    try:
        # Записываем все байты
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)
    
    # Ротация .bak
    if os.path.exists(path):
        try:
            os.replace(path, path + ".bak")
        except Exception:
            pass  # игнорируем ошибки backup
    
    os.replace(tmp, path)

def safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    """
    Безопасная загрузка JSON с верификацией:
    - Проверяет _magic
    - Верифицирует SHA256 контрольную сумму
    - Возвращает None при любых ошибках
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        
        data = json.loads(s)
        
        # Проверка magic
        if data.get("_magic") != MAGIC:
            return None
        
        # Верификация SHA256
        sha_stored = data.get("_sha256")
        if not sha_stored:
            return None
        
        data2 = dict(data)
        data2.pop("_sha256", None)
        
        # ИСПРАВЛЕНО: добавлен ensure_ascii=False
        body_verify = json.dumps(data2, ensure_ascii=False, sort_keys=True)
        chk = hashlib.sha256(body_verify.encode("utf-8")).hexdigest()
        
        if sha_stored != chk:
            return None
        
        return data
    except Exception:
        return None

def sane_vec(v: np.ndarray, max_abs: float = 1e3) -> bool:
    """
    Проверяет валидность numpy вектора:
    - Одномерный массив
    - Все значения конечные (не inf/nan)
    - Абсолютные значения в разумных пределах
    """
    if not isinstance(v, np.ndarray):
        return False
    return (v.ndim == 1) and np.isfinite(v).all() and (np.max(np.abs(v)) <= max_abs)

def sane_prob(p: float, allow_boundaries: bool = False) -> bool:
    """
    Проверяет валидность вероятности:
    - Конечное число
    - В диапазоне (0, 1) или [0, 1] если allow_boundaries=True
    """
    if not np.isfinite(p):
        return False
    if allow_boundaries:
        return 0.0 <= p <= 1.0
    else:
        return 0.0 < p < 1.0
