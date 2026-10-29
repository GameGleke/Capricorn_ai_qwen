"""Зрение Капри: скриншот всего экрана + анализ через ИИ-бэкенд, отчёт в файл.

Как пользоваться голосом: «посмотри на экран и опиши, что там», «что у меня открыто на экране»,
«проанализируй экран и сохрани отчёт». Результат — текстовый файл
%USERPROFILE%\\Documents\\Capri\\screen_ГГММДД_ЧЧСС.txt (плюс путь озвучивается).

Технически: PIL.ImageGrab делает снимок, он сжимается до ~1280 px по длинной стороне;
если бэкенд поддерживает vision (vision=True) — картинка уходит модели целиком;
иначе модель получает OCR-текст (pytesseract, если стоит) и описание окон, и пишет
структурный разбор. Всё аккуратно деградирует: без vision и без OCR — минимум фактов.
"""
import datetime
import logging
import os
import re

log = logging.getLogger("capri")


def capture(max_side=1280):
    """Скриншот всех мониторов → PIL.Image, уменьшенный для экономии токенов/VRAM."""
    from PIL import ImageGrab
    img = ImageGrab.grab(all_screens=True)
    w, h = img.size
    if max(w, h) > max_side:
        k = max_side / max(w, h)
        img = img.resize((int(w * k), int(h * k)))
    return img


def ocr_text(img):
    """Распознать текст на картинке (нужен pytesseract + Tesseract с rus+eng). Пустая строка, если нет."""
    try:
        import pytesseract
        lang = "rus+eng"
        try:
            if lang not in pytesseract.get_languages(config=""):
                lang = "eng"
        except Exception:
            pass
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception as e:
        log.info("OCR недоступен: %s", e)
        return ""


def open_windows(limit=25):
    """Список заголовков видимых окон — «что открыто» даже без распознавания картинки."""
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        titles = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def cb(hwnd, _):
            if u.IsWindowVisible(hwnd) and not u.IsIconic(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                u.GetWindowTextW(hwnd, buf, 256)
                t = buf.value.strip()
                if t and len(t) > 1:
                    titles.append(t)
            return True
        u.EnumWindows(cb, 0)
        return titles[:limit]
    except Exception:
        return []


PROMPT = ("Вот снимок рабочего стола пользователя {name} и то, что видно на нём. "
          "Сделай структурированный разбор на русском:\n"
          "1) ЧТО ОТКРЫТО — приложения, окна, вкладки.\n"
          "2) СОДЕРЖИМОЕ — ключевые надписи, сообщения, цифры, которые видны.\n"
          "3) ЧЕМ ЗАНЯТ ПОЛЬЗОВАТЕЛЬ — предположи по активности.\n"
          "4) ЗАМЕЧАНИЯ — что выглядит странно или требует внимания.\n"
          "Пиши коротко пунктами, без воды, без эмодзи. Не выдумывай того, чего нет в данных.\n\n")


def analyze_screen(backend, question=""):
    """Главная функция: возвращает (текст_отчёта, путь_к_файлу) или (None, причина)."""
    try:
        img = capture()
    except Exception as e:
        return None, f"не удалось сделать снимок экрана: {e}"
    facts = ["Окна на экране: " + "; ".join(open_windows()) or "Окон не видно"]
    text = ocr_text(img)
    if text:
        facts.append("Распознанный текст с экрана:\n" + text[:6000])
    else:
        facts.append("Точный текст с экрана распознать не удалось (OCR не установлен).")

    answer = None
    # 1) если модель умеет смотреть на картинки — отдаём ей сам снимок
    if backend is not None and getattr(backend, "vision", False):
        try:
            answer = backend.ask_image(img, (question or "Опиши и проанализируй, что происходит на этом экране.") +
                                       "\nСтруктура ответа: ЧТО ОТКРЫТО / СОДЕРЖИМОЕ / ЧЕМ ЗАНЯТ ПОЛЬЗОВАТЕЛЬ / ЗАМЕЧАНИЯ.")
        except Exception as e:
            log.info("vision-запрос не вышел: %s", e)
    # 2) иначе — текст+окна в обычный чат
    if not answer and backend is not None and getattr(backend, "ready", True):
        prompt = PROMPT.format(name="") + "\n\n".join(facts)
        if question:
            prompt += "\n\nПлюс отдельный вопрос пользователя: " + question
        try:
            answer = backend.generate([{"role": "system", "content":
                                        "Ты — глаз и аналитик. Отвечай строго по предоставленным данным, пунктами."},
                                       {"role": "user", "content": prompt[:15000]}], max_tokens=700)
        except Exception as e:
            log.info("чат-разбор экрана не вышел: %s", e)
    if not answer:
        answer = "ИИ-разбор недоступен. Сырые данные:\n\n" + "\n\n".join(facts)

    folder = os.path.join(os.path.expanduser("~"), "Documents", "Capri")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"screen_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    png = path[:-4] + ".png"
    try:
        img.save(png)
    except Exception:
        png = None
    body = ("Отчёт Капри по экрану. " + datetime.datetime.now().strftime("%d.%m.%Y %H:%M") +
            ("\nСнимок: " + png if png else "") + "\n\n" + answer.strip() + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return answer.strip(), path


# ---------- разбор голосовой команды ----------

SCREEN_Q_RE = re.compile(r"(что (у меня )?(сейчас )?(на экране|открыто)|опиши (экран|что видишь)|"
                         r"проанализируй экран|посмотри (на )?экран|видишь (экран|монитор|мой экран)|"
                         r"разбор экрана|скриншот и (анализ|разбор|описание))")


def is_screen_command(c):
    return bool(SCREEN_Q_RE.search(c))


def extract_question(c):
    """Из «посмотри на экран и скажи, много ли у меня ошибок» — вычленить хвост-вопрос."""
    m = re.search(r"(?:и|а также|затем)\s+(скажи|ответь|найди|проверь|опиши|расскажи)\s+(?P<q>.+)$", c)
    return m.group("q").strip() if m else ""
