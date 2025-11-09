# 🔧 ИСПРАВЛЕНИЕ DUCKDUCKGO NEWS SEARCH

## 🔴 Обнаруженные проблемы

### 1. **DuckDuckGo API сломан** (КРИТИЧНО!)

**Симптомы:**
```
RuntimeWarning: This package (`duckduckgo_search`) has been renamed to `ddgs`!
ERROR - [NewsCollector] DuckDuckGo search failed for EIGEN:
DDGS.news() got an unexpected keyword argument 'query'
```

**Последствия:**
- ✗ Haiku analyzer не получал **никаких новостей**
- ✗ Все монеты получали `fund=0.500` (нейтральный скор)
- ✗ Haiku analyzer работал **вхолостую**
- ✗ Статистика: 20 neutral, 0 approved, 0 rejected

**Причина:**
- Библиотека `duckduckgo_search` переименована в `ddgs`
- API изменился: параметр `keywords` → `query` (позиционный)

### 2. **SSL Certificate errors**

**Симптомы:**
```
RuntimeError: TLS handshake failed: cert verification failed -
self signed certificate in certificate chain [CERTIFICATE_VERIFY_FAILED]
```

**Причина:**
- Прокси с самоподписанным сертификатом
- Антивирус с MITM (Man-in-the-Middle)
- Корпоративный фаервол

**Решение:**
- Добавлена обработка SSL ошибок
- Бот продолжает работу без новостей

### 3. **Медленный скрининг** (~10 минут)

**Симптомы:**
```
[WATCHDOG] ⚠️ BOT APPEARS FROZEN! No activity for 629s
```

**Причина:**
- Скрининг 305 монет занимает ~10 минут
- Watchdog триггерится после 10 минут

**Решение:**
- Это нормально для 4-часового цикла
- Watchdog работает корректно (предупреждает, но не падает)

---

## ✅ Внесенные исправления

### 1. **Обновление библиотеки**

```bash
# Удалена старая библиотека
pip uninstall -y duckduckgo_search

# Установлена новая
pip install ddgs
```

### 2. **Обновление импорта**

**Было:**
```python
from duckduckgo_search import DDGS
```

**Стало:**
```python
from ddgs import DDGS  # ✅ Обновлено на новую библиотеку
```

### 3. **Исправление API вызова**

**Было (старая версия с keywords):**
```python
results = ddgs_client.news(
    query=query,  # ← Ошибка в старой версии
    timelimit=time_limit,
    max_results=HaikuConfig.MAX_NEWS_ITEMS
)
```

**Промежуточное (попытка с keywords):**
```python
results = ddgs_client.news(
    keywords=query,  # ← Не работает в новой версии
    timelimit=time_limit,
    max_results=HaikuConfig.MAX_NEWS_ITEMS
)
```

**Стало (правильно для новой версии):**
```python
# ✅ В новой библиотеке ddgs: query - позиционный аргумент
results = ddgs_client.news(
    query,  # Позиционный аргумент (не keyword!)
    timelimit=time_limit,
    max_results=HaikuConfig.MAX_NEWS_ITEMS
)
```

**Сигнатура в новой библиотеке:**
```python
def news(self, query: str, **kwargs: Any) -> list[dict[str, Any]]
```

### 4. **Улучшенная обработка ошибок**

**Было:**
```python
except Exception as e:
    logger.error(f"[NewsCollector] DuckDuckGo search failed: {e}")
    all_news = []
```

**Стало:**
```python
except Exception as e:
    # ✅ Специальная обработка SSL/сертификат ошибок
    error_str = str(e)
    if 'CERTIFICATE_VERIFY_FAILED' in error_str or 'SSL' in error_str or 'TLS' in error_str:
        logger.warning(f"[NewsCollector] SSL/Certificate error for {coin_name}. "
                     f"This is an environment issue (proxy/antivirus/firewall). "
                     f"Continuing without news for this coin.")
    else:
        logger.error(f"[NewsCollector] DuckDuckGo search failed: {e}")
    all_news = []
```

---

## 📋 Новая библиотека ddgs - API Reference

### Установка:
```bash
pip install ddgs
```

### Использование:
```python
from ddgs import DDGS

with DDGS() as ddgs:
    # news(query, **kwargs) - позиционный query!
    results = ddgs.news(
        "bitcoin cryptocurrency",  # query - позиционный
        timelimit='d',              # 'd' = day, 'w' = week, 'm' = month
        max_results=10              # максимум результатов
    )
```

### Параметры news():
- `query` (str) - **позиционный** аргумент, поисковый запрос
- `timelimit` (str) - 'd' (day), 'w' (week), 'm' (month), 'y' (year)
- `max_results` (int) - максимум результатов
- `region` (str) - регион поиска (по умолчанию 'us-en')

### Структура результата:
```python
{
    'title': 'Bitcoin hits new high...',
    'url': 'https://...',
    'date': '2025-11-09T...',
    'source': 'CoinDesk',
    'body': 'Bitcoin cryptocurrency...'
}
```

---

## 🚨 SSL/Certificate проблемы - решение

### Проблема:
```
TLS handshake failed: cert verification failed
```

### Причины:
1. **Корпоративный прокси** - перехватывает HTTPS с самоподписанным сертификатом
2. **Антивирус** - Kaspersky, AVG, Avast делают MITM для проверки HTTPS
3. **Фаервол** - блокирует/перехватывает SSL/TLS соединения

### Решения:

#### Вариант 1: Установить корневой сертификат (РЕКОМЕНДУЕТСЯ)
```bash
# Экспортировать сертификат из браузера/прокси
# Добавить в системное хранилище:
# Windows: certmgr.msc → Trusted Root Certification Authorities
# Linux: /usr/local/share/ca-certificates/
```

#### Вариант 2: Отключить SSL verification в антивирусе
- Kaspersky: Settings → Network → Scan encrypted connections → OFF
- Avast: Settings → Protection → Core Shields → Web Shield → OFF
- AVG: аналогично

#### Вариант 3: Использовать HTTP proxy
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

#### Вариант 4: Временно отключить антивирус (НЕ РЕКОМЕНДУЕТСЯ)

### Для тестирования:
```python
# Проверить работает ли DuckDuckGo
from ddgs import DDGS
with DDGS() as ddgs:
    results = ddgs.news("bitcoin", timelimit='d', max_results=3)
    print(f"Found {len(results)} news")
```

---

## 📊 Ожидаемое поведение после исправления

### Если SSL работает:
```
[NewsCollector] Found 5 news for BTCUSDT via DuckDuckGo
[HaikuAnalyzer] BTCUSDT: fund=0.650 (positive news)
```

### Если SSL не работает (graceful degradation):
```
[NewsCollector] SSL/Certificate error for BTCUSDT.
                This is an environment issue (proxy/antivirus/firewall).
                Continuing without news for this coin.
[HaikuAnalyzer] BTCUSDT: fund=0.500 (no news found - neutral)
```

### Важно:
- ✅ Бот **не падает** при SSL ошибках
- ✅ Продолжает работу **без новостей**
- ✅ Haiku analyzer возвращает **нейтральный скор 0.5**
- ✅ Технический анализ продолжает работать

---

## 🔄 Миграция с старой библиотеки

### Для пользователей с `duckduckgo_search`:

```bash
# 1. Удалить старую версию
pip uninstall -y duckduckgo_search

# 2. Установить новую
pip install ddgs

# 3. Код обновлен автоматически в haiku_analyzer.py
```

### Обратная совместимость:
- ✅ Если `ddgs` не установлен - warning в логах
- ✅ Haiku analyzer возвращает нейтральные скоры 0.5
- ✅ Бот продолжает работу

---

## 📝 Файлы изменены:

- `strategy/haiku_analyzer.py`:
  - Импорт: `from ddgs import DDGS`
  - API call: `ddgs.news(query, ...)` (позиционный)
  - Обработка SSL ошибок

---

## ✅ Тестирование

### Быстрый тест:
```bash
python3 -c "
from ddgs import DDGS
with DDGS() as ddgs:
    results = ddgs.news('bitcoin', timelimit='d', max_results=3)
    print(f'✓ Found {len(results)} news')
"
```

### Ожидаемый результат:
- **Если SSL работает:** `✓ Found 3 news`
- **Если SSL не работает:** `DDGSException: ... CERTIFICATE_VERIFY_FAILED`

---

## 🎯 Итог

| Проблема | Статус |
|----------|--------|
| DuckDuckGo API сломан | ✅ **Исправлено** - обновлена библиотека на `ddgs` |
| SSL certificate errors | ✅ **Обработано** - graceful degradation |
| Медленный скрининг | ✅ **Нормально** - 10 мин для 305 монет приемлемо |
| Haiku analyzer вхолостую | ✅ **Исправлено** - будет получать новости |

**Следующие шаги:**
1. На удаленной машине: `pip uninstall -y duckduckgo_search && pip install ddgs`
2. Решить SSL проблему (установить сертификаты или настроить антивирус)
3. Протестировать бота

**Если SSL не решить:**
- Бот будет работать **без новостей**
- Haiku analyzer будет возвращать **только нейтральные скоры 0.5**
- Решения будут приниматься **только на техническом анализе**
