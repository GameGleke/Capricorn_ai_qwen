"""Установка «разума» через llama.cpp: llama-cpp-python + GGUF-модель Qwen с Hugging Face.

    python tools/install_llama.py

Вызывается из setup_llama.bat. Никакой Ollama не нужен: llama.cpp сама тянет слои модели
на видеокарту (NVIDIA/AMD), а чего не влезло — считает на процессоре. Модель выбирается
по памяти ПК; потом в config.json прописывается "ai_backend": "llama".
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE, "config.json")
REPO = "unsloth/Qwen3.5-9B-GGUF"
# (файл GGUF, человекочитаемое описание, минимальная оперативка ГБ)
QUANTS = [
    ("Qwen3.5-9B-Q4_K_M.gguf", "умная, ~5.3 ГБ — нужна видеокарта от 6 ГБ или 16 ГБ памяти", 12),
    ("Qwen3.5-9B-Q3_K_M.gguf", "чуть глупее, ~4.4 ГБ — для ПК с 12 ГБ памяти", 10),
    ("Qwen3.5-4B-Q4_K_M.gguf", "лёгкая 4B, ~2.8 ГБ — ноутбуки и слабые ПК, отвечает проще", 8),
]


def say(text=""):
    print(text, flush=True)


def run(cmd):
    say("[Capri] $ " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def ram_gb():
    try:
        with open("/proc/meminfo") as f:  # Linux/macOS fallback
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) // 2**20
    except OSError:
        pass
    try:
        import ctypes
        k = ctypes.windll.kernel32
        class STAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = STAT(); s.dwLength = ctypes.sizeof(s)
        k.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullTotalPhys / 2**30
    except Exception:
        return 16  # не угадали — считаем средний ПК


def nvidia_vram_gb():
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return 0.0
    try:
        out = subprocess.run([nvsmi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max((float(x) / 1024 for x in out.split() if x.replace(".", "").isdigit()), default=0.0)
    except Exception:
        return 0.0


def pick_quant(vram, ram):
    """Самая умная квантизация, которая влезет: на GPU — по видеопамяти, без GPU — по RAM."""
    budget = vram - 1.5 if vram >= 6 else min(ram - 4, 10)  # запас на систему и контекст
    for fname, desc, need in QUANTS:
        size = {"Qwen3.5-9B-Q4_K_M.gguf": 5.3, "Qwen3.5-9B-Q3_K_M.gguf": 4.4,
                "Qwen3.5-4B-Q4_K_M.gguf": 2.8}[fname]
        if size <= budget and ram >= need:
            return fname
    return QUANTS[-1][0]


def edit_config(**kv):
    data = {}
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG, encoding="utf-8-sig") as f:
                data = json.load(f)
        except ValueError:
            say("[!] config.json с ошибкой JSON — правку настроек сделай вручную.")
            return
    data.update(kv)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def install_package():
    try:
        import llama_cpp  # noqa: F401
        say("[OK] llama-cpp-python уже установлена.")
        return True
    except ImportError:
        pass
    say("[Capri] Ставлю llama-cpp-python...")
    if run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"]):
        pass
    # обычный wheel компилируется на CPU — работает везде, GPU подхватывает, если собрана CUDA-версия
    if run([sys.executable, "-m", "pip", "install", "llama-cpp-python"]):
        return True
    say("[!] Не вышло. Попробуй так (компилятор должен быть установлен):")
    say("    python -m pip install llama-cpp-python --no-build-isolation")
    say("    Для NVIDIA быстрее: python -m pip install llama-cpp-python --config-settings=build=\"cmake\" "
        "--config-settings=arguments.cmake.build-type=Release (нужен Visual Studio Build Tools)")
    return False


def download(fname):
    local_dir = os.path.join(BASE, "gguf_models")
    os.makedirs(local_dir, exist_ok=True)
    dest = os.path.join(local_dir, fname)
    if os.path.isfile(dest) and os.path.getsize(dest) > 100_000_000:
        say(f"[OK] {fname} уже скачена ({os.path.getsize(dest)//2**20} МБ).")
        return True
    say(f"[Capri] Качаю {REPO}/{fname} (~{round(size_of(fname),1)} ГБ). Это один раз, дальше офлайн.")
    say("        Если huggingface.co недоступен — включи VPN или задай зеркало: set HF_ENDPOINT=https://hf-mirror.com")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, fname, local_dir=local_dir)
        return os.path.isfile(path)
    except ImportError:
        url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r, open(dest + ".part", "wb") as out:
                shutil.copyfileobj(r, out, length=2**20)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:
            say(f"[!] Скачивание не удалось: {e}")
            return False
    except Exception as e:
        say(f"[!] Скачивание не удалось: {e}")
        return False


def size_of(fname):
    for f, d, n in QUANTS:
        if f == fname:
            return float(d.split("~")[1].split(" ")[0])
    return 5.0


def test(fname):
    """Реальный тест: грузим модель и просим ответить одной фразой. Именно это раньше молча падало."""
    say("[Capri] Проверяю, что модель отвечает (первый запуск может занять пару минут)...")
    code = (
        "import sys; sys.path.insert(0, r'%s');\n"
        "from llama_cpp import Llama\n"
        "llm = Llama(model_path=r'%s', n_ctx=4096, n_gpu_layers=99, verbose=False)\n"
        "out = llm.create_chat_completion(messages=[{'role':'user','content':'Скажи одну короткую фразу на русском.'}], max_tokens=40)\n"
        "print('ОТВЕТ:', out['choices'][0]['message']['content'][:200])\n" % (BASE, os.path.join(BASE, "gguf_models", fname))
    )
    r = subprocess.run([sys.executable, "-c", code], timeout=900)
    return r.returncode == 0


def main():
    say("=== Установка разума Капри через llama.cpp (без Ollama) ===")
    ram = ram_gb()
    vram = nvidia_vram_gb()
    say(f"Память: RAM ~{int(ram)} ГБ" + (f", видеопамять NVIDIA ~{vram:.1f} ГБ" if vram else ", отдельной видеокарты NVIDIA не найдено (будет считать процессор — медленнее)"))
    if ram < 8:
        say("[!] Меньше 8 ГБ оперативки — даже лёгкая модель будет работать очень тяжело.")
    if not install_package():
        input("[Capri] Нажми Enter для выхода.")
        return 1
    fname = pick_quant(vram, ram)
    desc = next((d for f, d, n in QUANTS if f == fname), fname)
    say(f"[Capri] Выбираю {fname} — {desc}")
    if not download(fname):
        input("[Capri] Скачивание не удалось. Проверь интернет/VPN и нажми Enter.")
        return 1
    ok = test(fname)
    edit_config(local_ai=True, ai_backend="llama", llama_repo=REPO, llama_file=fname)
    say('[OK] В config.json записано: "ai_backend": "llama", "llama_file": "%s".' % fname)
    if ok:
        say("[OK] Модель отвечает. Запускай run.bat — Капри будет болтать через llama.cpp.")
    else:
        say("[!] Тестовый ответ не получен. Причина — выше в этом окне и в capri.log.")
        say("    Часто помогает выбор более лёгкой модели: удали gguf_models\\%s и перезапусти setup_llama.bat," % fname)
        say("    или впиши в config.json \"llama_file\": \"Qwen3.5-4B-Q4_K_M.gguf\".")
    say("\nГотово. Закрой это окно и запусти run.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
