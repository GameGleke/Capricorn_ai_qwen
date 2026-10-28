"""Живой поиск в интернете: Капри сама находит ответ и пересказывает его голосом.

Раньше на «свежие» вопросы (погода, новости, курсы) она только открывала Google и говорила
«спроси — загуглю». Теперь модуль сам делает запрос к DuckDuckGo (без ключей и API),
собирает заголовки и сниппеты и отдаёт их нейросети, которая отвечает по найденному.

Работает и без ddgs: fallback — обычный HTML-запрос к DuckDuckGo Lite через urllib.
"""
import datetime
import logging
import re
import urllib.parse
import urllib.request

log = logging.getLogger("capri")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# что считаем «свежим» — то, чего модель знать не может по определению
FRESH_Q = re.compile(r"погод|новост|курс (доллар|евро|рубл|валют|биткоин|крипт)|сколько стоит|цена|цены|"
                     r"счет матча|результат мат(?:ча|чей)|кто выиграл|расписани|пробк|курort|"
                     r"сегодня|сейчас|вчерашн|последни|новый сезон|премьер|выход фильма|"
                     r"трафик|котировк|акци|btc|eth", re.IGNORECASE)


def is_fresh(question):
    return bool(FRESH_Q.search(question or ""))


def _clean(s):
    s = urllib.parse.unquote(re.sub(r"<[^>]+>", "", s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    # DuckDuckGo иногда возвращает текст с url-обёрткой ссылки вида //duckduckgo.com/l/?uddg=...
    m = re.search(r"uddg=([^&]+)", s)
    if m:
        s = urllib.parse.unquote(m.group(1))
    return s[:400]


def web_search(query, max_results=5, timeout=8):
    """Список [{'title','url','snippet'}] или [] при любой ошибке (сеть молчит — не роняем Капри)."""
    try:
        from ddgs import DDGS
        with DDGS(timeout=timeout) as d:
            rows = d.text(query, region="ru-ru", safesearch="off", max_results=max_results) or []
        out = [{"title": _clean(r.get("title")), "url": r.get("href", ""),
                "snippet": _clean(r.get("body"))} for r in rows]
        out = [r for r in out if r["title"] or r["snippet"]]
        if out:
            return out
    except Exception as e:
        log.info("ddgs недоступен (%s), пробую DuckDuckGo Lite напрямую", e)
    try:  # резервный путь: POST на html/lite версию, без сторонних библиотек
        data = urllib.parse.urlencode({"q": query}).encode()
        req = urllib.request.Request("https://lite.duckduckgo.com/lite/", data=data,
                                     headers={"User-Agent": UA})
        html = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
        rows = []
        for block in re.findall(r'<a[^>]*class="result-link"[^>]*>(.*?)</a>(.*?)(?=result-link|$)',
                                html, re.S)[:max_results]:
            title = _clean(re.sub(r"<[^>]+>", " ", block[0]))
            sn_m = re.search(r'class="result-snippet"[^>]*>(.*?)</td>', block[1], re.S)
            url_m = re.search(r'href="([^"]+)"', block[0])
            snippet = _clean(sn_m.group(1)) if sn_m else ""
            if title or snippet:
                rows.append({"title": title, "url": urllib.parse.unquote(url_m.group(1)) if url_m else "",
                             "snippet": snippet})
        return rows
    except Exception as e:
        log.info("DuckDuckGo Lite тоже не ответил: %s", e)
        return []


def format_results(rows, budget=1600):
    """Привести выдачу к компактному тексту для промпта нейросети."""
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    lines = [f"Данные из интернета на {now}. Используй только их, не выдумывай."]
    used = len(lines[0])
    for i, r in enumerate(rows, 1):
        block = f"{i}) {r['title']}" + (f"\n   {r['snippet']}" if r["snippet"] else "")
        if used + len(block) > budget:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


ANSWER_SYSTEM = (
    "Ты — голосовой ассистент. Пользователь задал вопрос, требующий свежей информации из интернета. "
    "Ниже — результаты поиска. Ответь коротко (2-4 предложения), строго по фактам из результатов, "
    "простым разговорным русским, без списков, ссылок и упоминания поиска. "
    "Если точного ответа среди результатов нет — честно скажи, чего именно не нашлось, "
    "и подскажи, где посмотреть. Не выдумывай цифры."
)


def answer_with_web(question, ask_model, rows=None):
    """Найти в сети и ответить по найденному через переданную модель (ask(prompt_messages)).

    ask_model — функция, принимающая список сообщений [{role,content}...] и возвращающая текст.
    Возвращает строку-ответ или None, если ничего не найдено / модель не справилась.
    """
    rows = rows if rows is not None else web_search(question)
    if not rows:
        return None
    facts = format_results(rows)
    try:
        text = ask_model([{"role": "system", "content": ANSWER_SYSTEM + "\n\n" + facts},
                          {"role": "user", "content": question}])
    except Exception as e:
        log.info("веб-ответ через модель не вышел: %s", e)
        return None
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) >= 8 else None
