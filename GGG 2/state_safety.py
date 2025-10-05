# state_safety.py
import os, json, hashlib, tempfile, numpy as np

from typing import Optional, Dict, Any

MAGIC = "MLSTATE_v1"

def atomic_write_bytes(path: str, data: bytes):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        os.write(fd, data); os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_save_json(path, payload: dict):
    payload = dict(payload)
    # не затираем, если уже задано вызывающей стороной
    if "_magic" not in payload:
        payload["_magic"] = MAGIC
    payload["_version"] = 1
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload["_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    body2 = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", text=True)
    os.write(fd, body2.encode()); os.fsync(fd); os.close(fd)
    # ротация .bak
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)

def safe_load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        data = json.loads(s)
        if data.get("_magic") != MAGIC: return None
        sha = data.get("_sha256"); data2 = dict(data); data2.pop("_sha256", None)
        chk = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()
        if sha != chk: return None
        return data
    except Exception:
        return None

def sane_vec(v: np.ndarray, max_abs=1e3) -> bool:
    return (v.ndim==1) and np.isfinite(v).all() and (np.max(np.abs(v)) <= max_abs)

def sane_prob(p: float) -> bool:
    return np.isfinite(p) and (0.0 < p < 1.0)
