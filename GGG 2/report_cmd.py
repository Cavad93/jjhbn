# report_cmd.py
from __future__ import annotations
import time
import threading
from typing import Optional, Callable
import requests

from daily_report import compute_24h_metrics, render_report, try_send as try_send_daily

def start_report_listener(session: requests.Session,
                          token: str,
                          chat_id: int,
                          csv_path: str,
                          send_fn: Callable[..., bool],
                          poll_interval: int = 10) -> threading.Thread:
    """
    Лёгкий long-poll по Telegram getUpdates.
    При сообщении '/report' в нужном чате — отправляет суточный отчёт (без троттлинга).
    """
    base = f"https://api.telegram.org/bot{token}"

    def loop():
        offset: Optional[int] = None
        while True:
            try:
                resp = session.get(f"{base}/getUpdates",
                                   params={"timeout": 30, "offset": offset, "allowed_updates": ["message"]},
                                   timeout=40)
                data = resp.json() if resp.ok else {}
                if not data.get("ok"):
                    time.sleep(poll_interval)
                    continue

                for upd in data.get("result", []):
                    offset = int(upd["update_id"]) + 1
                    msg = upd.get("message") or {}
                    chat = msg.get("chat") or {}
                    if int(chat.get("id", 0)) != int(chat_id):
                        continue
                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue

                    if text.lower().startswith("/report"):
                        # формируем отчёт и шлём без суточного троттлинга
                        try:
                            # можно сразу использовать try_send(..., force=True)
                            txt = try_send_daily(csv_path, send_fn, force=True)
                            if not txt:
                                # защитный путь, если try_send не вернул текст
                                m = compute_24h_metrics(csv_path)
                                txt = render_report(m)
                                send_fn(txt, html=False, parse_mode="Markdown")
                        except Exception:
                            # проглатываем ошибки, чтобы не ронять поток
                            pass

            except Exception:
                time.sleep(3)

    t = threading.Thread(target=loop, name="ReportCmdListener", daemon=True)
    t.start()
    return t
