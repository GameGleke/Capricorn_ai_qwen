"""Ответы на вопросы через бесплатную модель Qwen 3.5 9B с Hugging Face в формате GGUF (llama.cpp).

Это «разум» без Ollama и без transformers: ставится одним setup_llama.bat, дальше работает
офлайн. Интерфейс совпадает с LocalAI/HFBackend: ask / chatter / summarize_report — main
подменяет бэкенды настройкой "ai_backend": "llama".

В отличие от старой HF-версии, все ошибки загрузки видны: stderr воркера пишется в capri.log,
а причина отказа показывается пользователю голосом и субтитром (раньше воркер умирал молча,
и Капри на любой вопрос отвечала «могу загуглить»).

Требуется: pip install llama-cpp-python huggingface_hub (ставит setup_llama.bat).
Тяжёлый инференс живёт в отдельном процессе tools/llama_worker.py — Qt-аватар не подвисает.
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
REPO_ID = os.environ.get("CAPRI_LLAMA_REPO", "unsloth/Qwen3.5-9B-GGUF")
FILENAME = os.environ.get("CAPRI_LLAMA_FILE", "Qwen3.5-9B-Q4_K_M.gguf")
NO_WINDOW = 0x08000000
LOAD_TIMEOUT = 2400   # первое включение: скачивание ~5 ГБ с Hugging Face может быть очень долгим
ASK_TIMEOUT = 300     # ответ на GPU обычно за секунды; на CPU для 9B — минуты


def _env():
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env.setdefault("HF_HOME", os.path.join(BASE, "hf_cache"))  # веса рядом с проектом
    return env


class LlamaBackend:
    """Тот же контракт, что у LocalAI: история, «поиск» при неуверенности, чистка ответа."""

    def __init__(self, cfg):
        self.repo = cfg.get("llama_repo", REPO_ID)
        self.file = cfg.get("llama_file", FILENAME)
        self.history = []
        self.ready = False
        self.error = ""          # почему не поднялась — main озвучивает это вместо тихого поиска
        self._proc = None
        self._lock = threading.Lock()
        self._id = 0
        self.on_model_change = None

    # ----- процесс-воркер -----

    def _spawn(self):
        exe = sys.executable
        worker = os.path.join(BASE, "tools", "llama_worker.py")
        try:
            self._proc = subprocess.Popen([exe, worker, self.repo, self.file],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                          bufsize=1, creationflags=NO_WINDOW if os.name == "nt" else 0,
                                          env=_env())
        except OSError as e:
            self.error = f"не удалось запустить python-воркер: {e}"
            return False
        self._out = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        return True

    def _pump_stdout(self):
        """Читает JSON-строки воркера в очередь — отдельно от потоков запросов."""
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._out.put(json.loads(line))
                except ValueError:
                    log.info("llama-воркер: мусорная строка: %s", line[:120])
        except Exception:
            pass
        self._out.put({"type": "dead"})

    def _pump_stderr(self):
        """Ошибки torch/llama.cpp больше не выбрасываются в DEVNULL — они в capri.log и в self.error."""
        try:
            for line in self._proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                log.info("llama-воркер: %s", line[:300])
                if line.startswith("ERROR:"):
                    self.error = line[6:].strip()
        except Exception:
            pass

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
                    self.error = msg["error"]
                    log.info("Ошибка llama-воркера: %s", msg["error"])
                    return None
                return msg

    # ----- публичный интерфейс (как у LocalAI) -----

    def installed(self):
        return [f"{self.repo}/{self.file}"]

    def running(self):
        return self.ready

    def start_server(self):
        """Поднимает воркер и грузит модель (первый раз — со скачиванием ~5 ГБ с Hugging Face)."""
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            self.error = ("llama-cpp-python не установлен — запусти setup_llama.bat "
                          "(или: python -m pip install llama-cpp-python)")
            log.info(self.error)
            return
        if not self._spawn():
            log.info(self.error or "не удалось запустить воркер")
            return
        msg = self._request({"cmd": "load"}, LOAD_TIMEOUT)
        if msg and msg.get("ok"):
            self.ready = True
            self.error = ""
            log.info("Модель %s (%s) через llama.cpp загружена", self.repo, self.file)
        else:
            reason = self.error or "модель не загрузилась (таймаут или не хватило памяти)"
            code = self._proc.poll() if self._proc else None
            self.shutdown()
            self.error = reason + (f" [код выхода воркера {code}]" if code not in (None, 0) else "")
            log.info("llama: %s", self.error)

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
            log.info("llama-модель недоступна: %s", e)
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
            system = f"{CHARACTER}\n\n{SPEECH_RULES}\n" + user_context()
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
            system = f"{CHARACTER}\n\n{SPEECH_RULES}\n" + user_context()
            return self._chat([
                {"role": "system", "content": system},
                {"role": "user", "content":
                 "Вот отчёт диагностики моего ПК. Скажи 2–4 предложения: общее состояние, главные "
                 "проблемы и что сделать. Безвредные ошибки вроде DistributedCOM не раздувай.\n\n"
                 + report[:12000]},
            ], max_tokens=400) or None
        except Exception:
            return None
