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
    ("Qwen3.5-9B-Q6_K.gguf", "самая умная, ~7.4 ГБ — для видеокарт от 10 ГБ (у тебя влезет целиком)", 16),
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
        size = {"Qwen3.5-9B-Q6_K.gguf": 7.4, "Qwen3.5-9B-Q4_K_M.gguf": 5.3,
                "Qwen3.5-9B-Q3_K_M.gguf": 4.4, "Qwen3.5-4B-Q4_K_M.gguf": 2.8}[fname]
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


def _pip_show(name):
    """Версия установленного через pip пакета (или None)."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "show", name],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 2**30
    except Exception:
        return 999.0


def nvidia_cuda_major():
    """Версия CUDA, которую поддерживает драйвер (старше или равна cu118 — вернёт None)."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        out = subprocess.run([nvsmi], capture_output=True, text=True, timeout=10).stdout
        import re as _re
        m = _re.search(r"CUDA Version:\s*(\d+)\.", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def install_prebuilt():
    """Ставим готовый CUDA-wheel llama-cpp-python со страницы релизов — компилятор не нужен.

    Важно: в релизных win-колесах тег cuXXX означает версию сборки, а pip видит обычный
    py3-none-win_amd64 и НЕ проверяет CUDA сам. Если у пользователя драйвер старее
    (например CUDA 11), колесо cu125 поставится, но не импортируется. Поэтому перебираем
    варианты от новой CUDA к старой и после каждой попытки проверяем реальный импорт.
    """
    drv = nvidia_cuda_major()
    order = ["cu132", "cu130", "cu125", "cu124", "cu123", "cu122", "cu121", "cu118"]
    if drv is not None:  # берём только теги не новее поддержки драйвера
        fit = [t for t in order if int(t[2:]) <= drv]
        if not fit:
            say("[!] Драйвер поддерживает только CUDA %s — готовые wheels llama.cpp (cu118+) не подойдут." % drv)
            return False
        order = fit
    tried = set()
    for cuda in order:
        for ver in ("0.3.36", "0.3.35", "0.3.34"):
            key = (ver, cuda)
            if key in tried:
                continue
            tried.add(key)
            tag = "v%s-%s" % (ver, cuda)
            fname = "llama_cpp_python-%s-py3-none-win_amd64.whl" % ver
            url = "https://github.com/abetlen/llama-cpp-python/releases/download/%s/%s" % (tag, fname)
            ok = _try_wheel(url, tag)
            if ok:
                return True
            if os.path.isfile(os.path.join(BASE, fname)):
                try:
                    os.remove(os.path.join(BASE, fname))
                except OSError:
                    pass
    say("[!] Ни одна готовая CUDA-сборка не подошла (см. сообщения выше).")
    return False


def _try_wheel(url, tag):
    # pip требует корректного имени файла wheel — сохраняем под именем из URL
    fname = url.split("/")[-1].split("?")[0] or "llama_cuda.whl"
    whl = os.path.join(BASE, fname)
    if os.path.isfile(whl) and open(whl, "rb").read(2) == b"PK":
        say("[Capri] Wheel %s уже скачен — пропускаю качание." % fname)
    else:
        if _disk_free_gb(os.environ.get("LOCALAPPDATA", BASE)) < 4:
            say("[!] Мало места на диске для CUDA-сборки (<4 ГБ) — пропуск, ставим CPU-вариант.")
            return False
        say("[Capri] Качаю готовую сборку llama.cpp %s (~500 МБ, один раз)..." % tag)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "capri-setup"})
            with urllib.request.urlopen(req, timeout=60) as r, open(whl + ".part", "wb") as out:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(2**20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        print("\r[Capri]   %3.0f%% (%.1f/%.1f МБ)" %
                              (done * 100 / total, done / 2**20, total / 2**20), flush=True)
            head = open(whl + ".part", "rb").read(2)
            if head != b"PK":
                raise RuntimeError("скачанный файл не является zip/wheel (начинается с %r) — вероятно, GitHub вернул HTML-страницу" % head)
            os.replace(whl + ".part", whl)
            say("")
        except Exception as e:
            say("[!] Готовый wheel не скачался: %s" % e)
            try:
                os.remove(whl + ".part")
            except OSError:
                pass
            return False
    ok = run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", whl])
    if not ok:
        try:
            os.remove(whl)
        except OSError:
            pass
        return False
    r = subprocess.run([sys.executable, "-c",
                        "import llama_cpp; from llama_cpp import Llama; print('[OK] импорт llama_cpp выполнен')"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        say("[OK] Установлена CUDA-версия llama-cpp-python (%s) — модель будет считать на видеокарте." % tag)
        try:
            os.remove(whl)
        except OSError:
            pass
        return True
    err = (r.stderr or "").strip()
    say("[!] Сборка %s поставилась, но не запускается Python-модуль: %s" % (tag, err[:300]))
    if "DLL" in err or "dll" in err:
        say("    Обычно это значит: версия CUDA сборки новее, чем поддерживает твой драйвер NVIDIA.")
        say("    Пробую сборку со старой CUDA...")
    return False


def _broken_install():
    """llama_cpp поставлен, но его llama.dll не грузится (сломанная/неподходящая сборка)."""
    try:
        import llama_cpp  # noqa: F401
        return False
    except ImportError:
        return False
    except Exception:
        return True


def install_package():
    if _broken_install():
        say("[!] Обнаружена сломанная установка llama-cpp-python (не грузится llama.dll).")
        say("[Capri] Удаляю её и ставлю готовую CUDA-сборку заново...")
        run([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
    else:
        try:
            import llama_cpp  # noqa: F401
            say("[OK] llama-cpp-python уже установлена и работает.")
            return True
        except ImportError:
            pass
    run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    vram = nvidia_vram_gb()
    if vram >= 6:  # есть NVIDIA — пробуем готовый CUDA-wheel без всякой компиляции
        if install_prebuilt():
            return True
        say("[Capri] Возвращаюсь к обычной установке...")
    say("[Capri] Ставлю llama-cpp-python (CPU-сборка)...")
    if run([sys.executable, "-m", "pip", "install", "llama-cpp-python"]):
        try:
            import llama_cpp  # noqa: F401
            return True
        except Exception as e:
            say("[!] CPU-сборка поставилась, но не запускается: %s" % str(e)[:200])
            say("    Удаляю её — дальше нужен готовый CUDA-wheel или Visual Studio Build Tools.")
            run([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
            return False
    # частая причина на Windows: нет компилятора (nmake/Visual Studio Build Tools)
    say("[!] Сборка из исходников не удалась — скорее всего, на Windows нет компилятора C/C++")
    say("    (ошибка 'nmake: no such file or directory' — это ровно оно).")
    if vram >= 6:
        say("    У тебя NVIDIA-видеокарта — попробуй ещё раз setup_llama.bat: он качает")
        say("    готовый CUDA-wheel (~500 МБ), компилятор для него НЕ нужен.")
        say("    Если и он не подошёл — обнови драйвер NVIDIA (нужна поддержка CUDA 12.5+):")
        say("    https://www.nvidia.com/Download/index.aspx")
    else:
        say("    Варианты без компилятора:")
        say("    1) Установить Visual Studio Build Tools (галочка Desktop development with C++) и повторить:")
        say("       https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/")
        say("    2) Временно: python tools/install_llama.py --cpu-only (если есть готовый wheel)")
        say("    3) Вернуться на Ollama: setup_ai.bat (там всё в одном установщике, без сборки).")
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
        "llm = Llama(model_path=r'%s', n_ctx=4096, n_gpu_layers=-1, verbose=False)\n"
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
