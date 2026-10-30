"""Диагностика «разума» Капри: почему вместо ответов она отправляет всё в поиск.

    python diagnose_ai.py

Проверяет config.json, наличие библиотек (transformers/torch или Ollama), скачанные веса и
делает реальный тестовый вопрос через выбранный бэкенд. В конце пишет вывод в ai_diag.txt —
сфоткай это окно или пришли файл, если что-то из этого непонятно.
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT = []


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def main():
    say("=== Диагностика ИИ Капри ===")
    say(f"Python: {sys.version.split()[0]} ({sys.executable})")

    # 1) настройки
    cfg = {}
    path = os.path.join(BASE, "config.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig") as f:
                text = f.read()
            cfg = json.loads(text) if text.strip() else {}
        except ValueError as e:
            say(f"[ОШИБКА] config.json сломан: {e}")
    else:
        say("[!] config.json нет — Капри работает со стандартными настройками.")
    backend = cfg.get("ai_backend", "ollama")
    say(f"local_ai   = {cfg.get('local_ai', True)}")
    say(f"ai_backend = {backend!r}")
    say(f"hf_model   = {cfg.get('hf_model', 'Qwen/Qwen3.5-9B')}")
    say(f"local_model= {cfg.get('local_model', 'qwen3:8b')}")
    if not cfg.get("local_ai", True):
        say("[!] local_ai выключен — нейросеть не используется вообще. Впиши в config.json \"local_ai\": true.")

    # 2) библиотека transformers
    hf_ok = False
    try:
        import transformers
        say(f"[OK] transformers {transformers.__version__}")
        hf_ok = True
    except ImportError:
        say("[!] transformers НЕ установлен — HF-бэкенд не может работать.")
    try:
        import torch
        cuda = torch.cuda.is_available()
        say(f"[OK] torch {torch.__version__}, CUDA доступна: {cuda}"
            + ("" if cuda else "  (генерация на CPU для 9B очень медленная!)"))
    except ImportError:
        say("[!] torch НЕ установлен.")

    # 3) веса модели
    model_id = cfg.get("hf_model", "Qwen/Qwen3.5-9B")
    cache = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    slug = "models--" + model_id.replace("/", "--")
    snapshots = os.path.join(cache, "hub", slug, "snapshots")
    have = os.path.isdir(snapshots) and any(
        os.path.isfile(os.path.join(dp, fn)) for dp, _, files in os.walk(snapshots)
        for fn in files if fn == "config.json")
    local_dir = os.path.join(BASE, "hf_models", model_id.replace("/", "--"))
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        say(f"[OK] веса лежат локально: {local_dir}")
    elif have:
        say(f"[OK] веса скачаны в кэш: {snapshots}")
    elif hf_ok:
        say(f"[!] весов {model_id} нигде нет — при запуске Капри будет качать ~6 ГБ молча, "
            "в фоне. Проще докачать заранее: python tools/install_hf.py")

    # 4) llama.cpp (GGUF-бэкенд) — проверяем библиотеку и скачанный .gguf
    if backend == "llama":
        try:
            import llama_cpp
            say(f"[OK] llama-cpp-python {getattr(llama_cpp, '__version__', '?')} установлена")
        except ImportError:
            say("[!] llama-cpp-python НЕ установлена — запусти setup_llama.bat.")
        repo = cfg.get("llama_repo", "unsloth/Qwen3.5-9B-GGUF")
        fname = cfg.get("llama_file", "Qwen3.5-9B-Q4_K_M.gguf")
        gguf_dir = os.path.join(BASE, "gguf_models")
        found = None
        for cand in (os.path.join(gguf_dir, fname),):
            if os.path.isfile(cand):
                found = cand
        if not found and os.path.isdir(gguf_dir):
            for d in os.listdir(gguf_dir):
                p = os.path.join(gguf_dir, d, fname)
                if os.path.isfile(p):
                    found = p
        if found:
            say(f"[OK] веса GGUF на месте: {found} ({os.path.getsize(found)//2**20} МБ)")
        else:
            say(f"[!] файла {fname} из {repo} нет в gguf_models\\ — первый запуск Капри будет качать его ~5 ГБ; "
                "проще докачать заранее: python tools/install_llama.py")

    # 5) Ollama (если выбран её бэкенд)
    if backend not in ("hf", "llama"):
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
                names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
            say(f"[OK] Ollama отвечает, модели: {names or 'НИ ОДНОЙ — запусти setup_ai.bat'}")
        except Exception:
            say("[!] Ollama не отвечает на 127.0.0.1:11434 — установи/запусти её или перейди на llama/hf.")

    # 6) живой тест: задать вопрос тому бэкенду, который выберет Капри
    if backend == "llama":
        from llama_backend import LlamaBackend
        say("\nТестирую llama.cpp-бэкенд (первый запуск может качать weights долго)...")
        be = LlamaBackend(cfg)
        be.start_server()
        if not be.ready:
            say(f"[ОШИБКА] модель не загрузилась — именно поэтому Капри шлёт всё в поиск.")
            say(f"         Причина: {be.error or 'неизвестна, смотри capri.log'}")
            say("         Частые причины: нет интернета на время скачивания ~5 ГБ, не хватает памяти,")
            say("         huggingface.co перекрыт провайдером (поможет VPN или HF_ENDPOINT=https://hf-mirror.com).")
        else:
            for q in ("Почему небо голубое?", "Расскажи стишок про котов"):
                a = be.ask(q)
                say(f"Вопрос: {q}\nОтвет: {a if a else '[пусто -> Капри откроет поиск]'}")
        be.shutdown()
    elif backend == "hf" and hf_ok:
        from hf_backend import HFBackend
        say("\nТестирую HF-бэкенд (первый запуск может качать weights долго)...")
        be = HFBackend(cfg)
        be.start_server()
        if not be.ready:
            say("[ОШИБКА] модель не загрузилась — именно поэтому Капри шлёт всё в поиск.")
            say("         Подробности в capri.log; чаще всего: нет интернета на время скачивания")
            say("         или не хватает памяти (нужно ~8 ГБ VRAM или ~16 ГБ RAM).")
        else:
            for q in ("Почему небо голубое?", "Расскажи стишок про котов"):
                a = be.ask(q)
                say(f"Вопрос: {q}\nОтвет: {a if a else '[пусто -> Капри откроет поиск]'}")
        be.shutdown()
    elif backend not in ("hf", "llama"):
        from local_ai import LocalAI
        say("\nТестирую Ollama-бэкенд...")
        la = LocalAI(cfg)
        la.start_server()
        a = la.ask("Почему небо голубое?")
        say(f"Модель: {la.model}\nОтвет: {a if a else '[пусто -> Капри откроет поиск]'}")

    with open(os.path.join(BASE, "ai_diag.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    say("\nВывод сохранён в ai_diag.txt")


if __name__ == "__main__":
    main()
