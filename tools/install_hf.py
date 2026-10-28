"""Установка «разума с Hugging Face» для Капри: transformers + Qwen 3.5 9B (~6 ГБ).

    python tools/install_hf.py [Qwen/Qwen3.5-9B]

Вызывается из setup_hf.bat. Ставит библиотеки, скачивает веса в кэш Hugging Face и
прописывает в config.json: "local_ai": true, "ai_backend": "hf". Если Ollama уже стоит
и устраивает — этот скрипт не нужен, достаточно поменять ai_backend в config.json.
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE, "config.json")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"


def say(text=""):
    print(text, flush=True)


def pip_install(*packages):
    say(f"[Capri] Устанавливаю: {' '.join(packages)} ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", *packages])
    return r.returncode == 0


def has_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max((int(x) for x in out.split()), default=0) / 1024
    except Exception:
        return 0


def ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:
        return 0


def save_config():
    data = {}
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG, encoding="utf-8-sig") as f:
                text = f.read()
            data = json.loads(text) if text.strip() else {}
        except ValueError:
            say("[Capri] В config.json ошибка — впиши вручную:")
            say('        "local_ai": true, "ai_backend": "hf",')
            say(f'        "hf_model": "{MODEL}",')
            return False
    data["local_ai"] = True
    data["ai_backend"] = "hf"
    data["hf_model"] = MODEL
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def main():
    vram, ram = has_gpu(), ram_gb()
    say(f"[Capri] Видеокарта NVIDIA: {f'{vram:.0f} ГБ' if vram else 'нет'}, оперативная память: {ram:.0f} ГБ")
    if vram < 8 and ram < 16:
        say("[Capri] Внимание: Qwen 9B требует ~8 ГБ видеопамяти ИЛИ ~16-18 ГБ обычной памяти.")
        say("        На слабом ПК лучше оставить Ollama и модель qwen3:4b (setup_ai.bat).")
        if input("[Capri] Всё равно ставим HF? (y/N): ").strip().lower() not in ("y", "д", "да"):
            return 1
    if not pip_install("transformers>=4.51", "accelerate"):
        say("[Capri] Библиотеки не установились. Проверь интернет/VPN и повтори.")
        return 1
    if not vram and not pip_install("torch", "--index-url", "https://download.pytorch.org/whl/cpu"):
        say("[Capri] Не удалось поставить torch на CPU. Скачай его сам: https://pytorch.org/get-started/locally/")
        return 1
    say(f"\n[Capri] Скачиваю модель {MODEL} с Hugging Face (~6 ГБ). Это один раз, потом офлайн.")
    say("        Если загрузка встаёт — включи VPN и повтори setup_hf.bat.\n")
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1")
    dl = subprocess.run([sys.executable, "-c",
                         f"from huggingface_hub import snapshot_download;"
                         f"snapshot_download('{MODEL}')"], env=env)
    if dl.returncode != 0:
        say("\n[Capri] Модель не скачалась. Проверь интернет/VPN и запусти setup_hf.bat ещё раз.")
        return 1
    if save_config():
        say(f"\n[Capri] Готово! Разум Капри — {MODEL} с Hugging Face (прописано в config.json).")
    say("[Capri] Перезапусти Капри (значок в трее -> «Выход», затем run.bat).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
