# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
import importlib.util
import types
import pytest

def _try_import_by_path(mod_name: str, file_path: Path):
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    return None

@pytest.fixture(scope="session")
def project():
    """
    Универсальный импорт проекта:
      1) Обычный import bnbusdrt6/meta_cem_mc
      2) Поиск рядом (если тесты лежат внутри репо)
      3) Фоллбеки: prob_calibrators / performance_metrics / delta_daily
         - если нет отдельных модулей, берём классы из bnbusdrt6 (если есть)
    """
    mod = types.SimpleNamespace()

    # 1) Пробуем стандартный импорт
    try:
        import bnbusdrt6 as _main
        mod.bnbusdrt6 = _main
    except Exception:
        _main = None

    try:
        import meta_cem_mc as _meta
        mod.meta_cem_mc = _meta
    except Exception:
        _meta = None

    # 2) Поиск рядом
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "bnbusdrt6.py",
        here.parent.parent / "bnbusdrt6.py",
        here.parent / "meta_cem_mc.py",
        here.parent.parent / "meta_cem_mc.py",
        here.parent / "prob_calibrators.py",
        here.parent.parent / "prob_calibrators.py",
        here.parent / "performance_metrics.py",
        here.parent.parent / "performance_metrics.py",
        here.parent / "delta_daily.py",
        here.parent.parent / "delta_daily.py",
    ]

    if _main is None:
        for p in candidates:
            if p.name == "bnbusdrt6.py":
                m = _try_import_by_path("bnbusdrt6", p)
                if m:
                    mod.bnbusdrt6 = m
                    break
    if _meta is None:
        for p in candidates:
            if p.name == "meta_cem_mc.py":
                m = _try_import_by_path("meta_cem_mc", p)
                if m:
                    mod.meta_cem_mc = m
                    break

    # 2.5) Пытаемся подтянуть вспомогательные как отдельные файлы
    for name in ["prob_calibrators", "performance_metrics", "delta_daily"]:
        if not hasattr(mod, name):
            for p in candidates:
                if p.name == f"{name}.py":
                    m = _try_import_by_path(name, p)
                    if m:
                        setattr(mod, name, m)
                        break

    # 3) Фоллбеки из bnbusdrt6
    if not hasattr(mod, "prob_calibrators"):
        ns = {}
        for cls in ["LogisticCalibrator", "IsotonicCalibrator"]:
            if hasattr(mod.bnbusdrt6, cls):
                ns[cls] = getattr(mod.bnbusdrt6, cls)
        if ns:
            ns.setdefault("HAVE_SK", True)
            mod.prob_calibrators = types.SimpleNamespace(**ns)

    if not hasattr(mod, "performance_metrics"):
        if hasattr(mod.bnbusdrt6, "PerfMonitor"):
            mod.performance_metrics = types.SimpleNamespace(
                PerfMonitor=getattr(mod.bnbusdrt6, "PerfMonitor")
            )

    if not hasattr(mod, "delta_daily"):
        if hasattr(mod.bnbusdrt6, "DeltaDaily"):
            mod.delta_daily = types.SimpleNamespace(
                DeltaDaily=getattr(mod.bnbusdrt6, "DeltaDaily")
            )

    assert hasattr(mod, "bnbusdrt6"), "Не найден bnbusdrt6.py"
    assert hasattr(mod, "meta_cem_mc"), "Не найден meta_cem_mc.py"
    return mod
