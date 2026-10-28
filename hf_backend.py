"""Ответы на вопросы через бесплатную модель Qwen 3.5 9B с Hugging Face (transformers).

Работает офлайн после первого скачивания весов (~6 ГБ в кэш HF), без ключей и подписок.
Интерфейс совпадает с LocalAI (Ollama): ask / chatter / summarize_report — main подменяет
ими один бэкенд другим настройкой "ai_backend": "hf".

Требуется: pip install transformers accelerate (GPU) или torch CPU-сборка (ноутбук без видеокарты).
Тяжёлый инференс живёт в отдельном процессе tools/hf_worker.py — так Qt-аватар не подвисает,
а CUDA-контекст не мешает основному процессу.
"""
import datetime
import json
import logging
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time

from persona import CHARACTER, SPEECH_RULES, user_context
from local_ai import (CHATTER_TOPICS, FRESH, SEARCH_MARK, FOREIGN, drop_third_person,
                      naturalize, drop_repeats, strip_reasoning)

log = logging.getLogger("capri")

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get("CAPRI_HF_MODEL", "Qwen/Qwen3.5-9B")
NO_WINDOW = 0x08000000
LOAD_TIMEOUT = 1200   # первое включение: скачивание ~6 ГБ с hf-cache может быть долгим
ASK_TIMEOUT = 180     # ответ на вопрос на GPU обычно за секунды; на CPU — дольше


def _env():
    """Как у основного процесса, но без PYTHONPATH проекта: воркер запускается системным
    python из venv, а переменные PYTHON* могут указывать на другой интерпретатор."""
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


class HFBackend:
    """Тот же контракт, что у LocalAI: история, «поиск» при неуверенности, чистка ответа."""

    def __init__(self, cfg):
        self.model = cfg.get("hf_model", MODEL_ID)
        self.history = []
        self.ready = False
        self._proc = None
        self._lock = threading.Lock()
        self._id = 0
        self.on_model_change = None

    # ----- процесс-воркер -----

    def _spawn(self):
        exe = sys.executable
        worker = os.path.join(BASE, "tools", "hf_worker.py")
        self._proc = subprocess.Popen([exe, worker, self.model], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      text=True, encoding="utf-8", bufsize=1,
                                      creationflags=NO_WINDOW if os.name == "nt" else 0,
                                      env=_env())
        thread = threading.Thread(target=self._pump_stdout, daemon=True)
        thread.start()
        self._out = queue.Queue()

    def _pump_stdout(self):
        """Читает JSON-строки воркера в очередь — отдельно от потоков запросов, чтобы
        два потока не блокировались на readline разных ответов."""
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._out.put(json.loads(line))
                except ValueError:
                    log.info("HF-воркер: мусорная строка: %s", line[:120])
        except Exception:
            pass
        self._out.put({"type": "dead"})

    def _request(self, payload, timeout):
        """Отправляет запрос воркеру и ждёт ответ с тем же id. None — если воркер умер/не успел."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return None
            self._id += 1
            rid = self._id
            payload = dict(payload, id=rid)
            try:
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except OSError:
                return None
            deadline = time.time() + timeout
            while True:
                left = deadline - time.time()
                if left <= 0:
                    return None
                try:
                    msg = self._out.get(timeout=min(left, 1.0))
                except queue.Empty:
                    continue
                if msg.get("type") == "dead":
                    self._proc = None
                    return None
                if msg.get("id") != rid:
                    continue  # просроченный ответ прежнего запроса
                if msg.get("error"):
                    log.info("Ошибка HF-воркера: %s", msg["error"])
                    return None
                return msg

    # ----- публичный интерфейс (как у LocalAI) -----

    def installed(self):
        """None — не проверяем; модель считается доступной, если воркер смог её поднять."""
        return [self.model]

    def running(self):
        return self.ready

    def start_server(self):
        """Поднимает воркер и грузит модель (первый раз — со скачиванием весов с Hugging Face)."""
        try:
            import transformers  # noqa: F401
        except ImportError:
            log.info("Нет transformers — установи: pip install transformers accelerate")
            return
        self._spawn()
        msg = self._request({"cmd": "load"}, LOAD_TIMEOUT)
        if msg and msg.get("ok"):
            self.ready = True
            log.info("Модель %s с Hugging Face загружена", self.model)
        else:
            self.shutdown()
            log.info("Не удалось загрузить %s — проверь интернет (первое скачивание ~6 ГБ) "
                     "и память (нужно ~7–8 ГБ)", self.model)

    def shutdown(self):
        proc, self._proc = self._proc, None
        self.ready = False
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def _chat(self, messages, max_tokens=300):
        if not self.ready:
            return ""
        msg = self._request({"cmd": "generate", "messages": messages, "max_new_tokens": max_tokens},
                            ASK_TIMEOUT)
        if not msg:
            return ""
        text = strip_reasoning(msg.get("text", ""))
        if text.endswith("[truncated]"):
            text = text[:-len("[truncated]")].rstrip()
        return text

    def ask(self, question):
        """Текст ответа или None, если лучше открыть поиск (свежие факты / модель не справилась)."""
        if FRESH.search(question.lower()):
            return None
        now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        self.history.append({"role": "user", "content": question})
        self.history = self.history[-6:]
        system = (f"{CHARACTER}\n\n{SPEECH_RULES}\n"
                  "Отвечай на любые вопросы о науке, истории, технике, языке, о себе, советы и просьбы рассказать что-то.\n"
                  f"Только если вопрос про текущие события (погода сейчас, свежие новости, курсы валют, цены,\n"
                  f"результаты недавних матчей), ответь ровно одним словом: {SEARCH_MARK}")
        messages = ([{"role": "system", "content": system + "\n" + user_context() + f"\nСейчас {now}."}]
                    + self.history)
        try:
            answer = self._chat(messages)
            if FOREIGN.search(answer):  # Qwen иногда срывается на китайский — пробуем ещё раз
                answer = self._chat(messages + [{"role": "system", "content":
                                    "Ответь заново, строго только русскими буквами, без иероглифов."}])
            answer = re.sub(r"\s+", " ", FOREIGN.sub("", answer)).strip()
        except Exception as e:
            log.info("HF-модель недоступна: %s", e)
            self.history.pop()
            return None
        if not answer:
            self.history.pop()
            return None
        recent = [m["content"] for m in self.history if m["role"] == "assistant"]
        answer = naturalize(drop_repeats(drop_third_person(answer), self.history), question, recent)
        answer = re.sub(r"\s*\([^)]*\)", "", answer)                      # ремарки в скобках
        answer = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", answer)  # эмодзи
        answer = re.sub(r"\s+", " ", answer).strip()
        answer = answer[:1].upper() + answer[1:]
        broken = len(answer) < 40 and not re.search(r"[.!?…]$", answer)
        if not answer or broken or SEARCH_MARK in answer.upper()[:20]:
            self.history.pop()
            return None
        self.history.append({"role": "assistant", "content": answer})
        return answer

    CHATTER_HINT = ("Ты сама решила ненадолго появиться на экране. Скажи одну короткую спонтанную фразу, "
                    "до пятнадцати слов. Не спрашивай, чем помочь, и не упоминай команды.")

    def chatter(self):
        if not self.ready:
            return None
        try:
            system = (f"{CHARACTER}\n\n{SPEECH_RULES}\n" + user_context())
            text = self._chat([{"role": "system", "content": system},
                               {"role": "user", "content":
                                self.CHATTER_HINT + f" Тема: {random.choice(CHATTER_TOPICS)}."}],
                              max_tokens=60)
        except Exception:
            return None
        text = drop_third_person(re.sub(r"\s+", " ", FOREIGN.sub("", text)).strip().strip('"«»'))
        text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", text).strip()
        return text if 5 < len(text) < 160 and re.search(r"[.!?…]$", text) else None

    def summarize_report(self, report):
        if not self.ready:
            return None
        try:
            system = (f"{CHARACTER}\n\n{SPEECH_RULES}\n" + user_context())
            return self._chat([
                {"role": "system", "content": system},
                {"role": "user", "content":
                 "Вот отчёт диагностики моего ПК. Скажи 2–4 предложения: общее состояние, главные "
                 "проблемы и что сделать. Безвредные ошибки вроде DistributedCOM не раздувай.\n\n"
                 + report[:12000]},
            ], max_tokens=400) or None
        except Exception:
            return None
