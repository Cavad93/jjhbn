# env_check.py — быстрая проверка загрузки .env
import os
try:
    from dotenv import load_dotenv
except Exception as e:
    print("Install python-dotenv first: pip install python-dotenv")
    raise

load_dotenv()
token = os.getenv("TG_BOT_TOKEN", "")
chat  = os.getenv("TG_CHAT_ID", "")
print("TG_BOT_TOKEN:", (token[:6] + "..." if token else "<empty>"))
print("TG_CHAT_ID  :", chat if chat else "<empty>")
print("WEB3_RPC    :", os.getenv("WEB3_RPC", "<empty>"))
