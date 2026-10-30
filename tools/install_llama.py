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


def nvidia_cuda_versions():
    """Все версии CUDA, которые видит система, как список float по убыванию.

    Берём и «CUDA Version» из nvidia-smi (максимум по ДРАЙВЕРУ), и реально
    установленные тулкиты CUDA Toolkit (переменная окружения CUDA_PATH, папки
    C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vXX.Y, PATH). Это важно:
    готовые wheels llama.cpp собирают с динамической привязкой к cuda.dll конкретной
    мажорной версии — wheel cu132 требует CUDA-тулкит 13.2 на машине даже при свежем
    драйвере. Раньше учитывался только драйвер, из-за чего ставилось колесо, чей
    llama.dll не мог найти свою cuda.dll (ошибка 'Could not find module ... llama.dll').
    """
    found = set()
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            out = subprocess.run([nvsmi], capture_output=True, text=True, timeout=10).stdout
            import re as _re
            for m in _re.finditer(r"CUDA Version:\s*(\d+)\.(\d+)", out):
                found.add(float("%s.%s" % (m.group(1), m.group(2))))
        except Exception:
            pass
    # реальные тулкиты: CUDA_PATH + стандартные папки + PATH
    import glob as _glob
    import re as _re
    cands = []
    cp = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_PATH_V_ANY")
    if cp:
        cands.append(cp)
    cands += _glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
    for p in (os.environ.get("PATH") or "").split(os.pathsep):
        if "NVIDIA GPU Computing Toolkit" in p or _re.search(r"[\\/]cuda[\\/_]?v?\d", p, _re.I):
            cands.append(p)
    for c in cands:
        m = _re.search(r"v?(\d+)\.(\d+)", c or "")
        if m:
            try:
                found.add(float("%s.%s" % (m.group(1), m.group(2))))
            except ValueError:
                pass
    return sorted(found, reverse=True)


def gh_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "capri-setup",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def find_prebuilt_wheel(cudas):
    """Ищем в релизах abetlen/llama-cpp-python готовые win-колёса под CUDA машины.

    Возвращает СПИСОК кандидатов [(download_url, имя_файла, тег_сборки), ...] от самой
    новой CUDA к самой старой (не более 4 штук — качать по ~500 МБ «вслепую» дорого).
    Учитываются ВСЕ версии CUDA, что видит система (тулкиты + драйвер): wheels llama.cpp
    динамически привязаны к cuda.dll конкретной мажорной версии, поэтому колесо cu132
    работает только если на машине реально есть CUDA 13.x — одного свежего драйвера мало.
    Список релизов берётся с GitHub API, поэтому имена файлов и URL никогда не устареют.
    """
    order = [132, 130, 128, 126, 125, 124, 123, 122, 121, 118]
    allowed = {int(round(c * 10)) for c in cudas} if cudas else set(order)
    # допускаем сборки той же мажорной версии, что любой найденный тулкит/драйвер
    majors = {c // 10 for c in allowed}
    order = [cu for cu in order if cu in allowed or (cu // 10) in majors and cu <= max(allowed, default=999)]
    if not order:
        say("[!] Не нашёл ни одной поддерживаемой версии CUDA (нужна минимум 11.8).")
        say("    Установи CUDA Toolkit или обнови драйвер NVIDIA: https://www.nvidia.com/Download/index.aspx")
        return []
    try:
        releases = gh_json("https://api.github.com/repos/abetlen/llama-cpp-python/releases?per_page=100")
    except Exception as e:
        say("[!] Не смог получить список релизов с GitHub: %s" % e)
        return []
    by_tag = {r["tag_name"]: r for r in releases}
    vers = sorted({t.split("-")[0][1:] for t in by_tag if "-cu" in t}, reverse=True)[:3]
    out = []
    seen = set()
    for cu in order:
        for ver in vers:
            rel = by_tag.get("v%s-cu%s" % (ver, cu))
            if not rel:
                continue
            for a in rel.get("assets", []):
                if a["name"].endswith("win_amd64.whl") and a["name"] not in seen:
                    seen.add(a["name"])
                    out.append((a["browser_download_url"], a["name"], "v%s-cu%s" % (ver, cu)))
    return out[:4]


def install_prebuilt():
    """Ставим готовый CUDA-wheel llama-cpp-python со страницы релизов — компилятор не нужен.

    Перебираем подходящие сборки от новой CUDA к старой; неподходящая (её llama.dll не
    грузится из-за отсутствия нужного cuda.dll) сносится автоматически, качается следующая.
    """
    cudas = nvidia_cuda_versions()
    if cudas:
        say("[Capri] Вижу CUDA в системе: %s." % ", ".join(str(c) for c in cudas))
    else:
        say("[Capri] Версии CUDA не найдены — буду пробовать самые свежие готовые сборки по очереди.")
    cands = find_prebuilt_wheel(cudas)
    if not cands:
        return False
    for url, fname, tag in cands:
        if _try_wheel(url, tag, fname):
            return True
    say("[!] Ни одна готовая CUDA-сборка не запустилась. Обычно помогает:")
    say("    1) Обновить драйвер NVIDIA И установить CUDA Toolkit нужной версии:")
    say("       https://developer.nvidia.com/cuda-downloads (выбери свою Windows x86_64)")
    say("    2) Либо вернуться на Ollama: setup_ai.bat (сам скачает всё необходимое, без wheels).")
    return False


def _try_wheel(url, tag, fname=None):
    # pip требует корректного имени файла wheel — сохраняем под именем из релиза
    fname = fname or url.split("/")[-1].split("?")[0] or "llama_cuda.whl"
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
    run([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
    try:
        os.remove(whl)
    except OSError:
        pass
    if "DLL" in err or "dll" in err:
        say("    Её llama.dll не нашёл нужную cuda.dll — этой сборке нужна своя версия CUDA Toolkit.")
        say("    Буду пробовать следующую сборку (или установи CUDA Toolkit: https://developer.nvidia.com/cuda-downloads).")
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


WHEELS_CPU = [
    # Готовые CPU-колёса llama-cpp-python с GitHub-релизов (тег без суффикса = CPU):
    # работают БЕЗ CUDA Toolkit и БЕЗ компилятора. Проверка импортом обязательна.
    ("https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/llama_cpp_python-0.3.36-py3-none-win_amd64.whl",
     "v0.3.36 cpu"),
    ("https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-win_amd64.whl",
     "v0.3.35 cpu"),
    ("https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34/llama_cpp_python-0.3.34-py3-none-win_amd64.whl",
     "v0.3.34 cpu"),
]


def _import_check(tag):
    """Реальная проверка, что библиотека грузится; при провале — удаление пакета."""
    try:
        import llama_cpp  # noqa: F401
        say(f"[OK] llama-cpp-python работает ({tag}).")
        return True
    except Exception as e:
        say(f"[!] Сборка {tag} поставилась, но не грузится: {str(e)[-160:]}")
        run([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
        return False


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def install_cpu_prebuilt():
    """Ставим готовые CPU-колёса (без CUDA Toolkit и без компилятора)."""
    for url, tag in WHEELS_CPU:
        fname = url.rsplit("/", 1)[-1]
        local = os.path.join(BASE, fname)
        if os.path.isfile(local) and open(local, "rb").read(2) != b"PK":
            _rm(local)  # мусорный файл вместо wheel — качаем заново
        if not os.path.isfile(local):
            say(f"[Capri] Качаю готовую CPU-сборку {tag} (~8 МБ, без CUDA Toolkit)...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "capri-setup"})
                with urllib.request.urlopen(req, timeout=60) as r, open(local + ".part", "wb") as out:
                    shutil.copyfileobj(r, out, length=2**20)
                if open(local + ".part", "rb").read(2) != b"PK":
                    raise RuntimeError("это не wheel (вернулась HTML-страница)")
                os.replace(local + ".part", local)
            except Exception as e:
                say("[!] Не скачалось: %s" % str(e)[:150])
                _rm(local + ".part")
                continue
        say(f"[Capri] Ставлю {fname}...")
        if not run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                    "--no-deps", local]):
            _rm(local)
            continue
        if _import_check(tag):
            _rm(local)
            return True
        _rm(local)  # битое колесо выбрасываем, чтобы pip не подставлял его снова
    return False


def install_package():
    if _broken_install():
        say("[!] Обнаружена сломанная установка llama-cpp-python (не грузится llama.dll).")
        say("[Capri] Удаляю её и ставлю рабочую сборку заново...")
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
    if vram >= 6:  # есть NVIDIA — пробуем готовый CUDA-wheel (максимальная скорость)
        if install_prebuilt():
            return True
        say("[Capri] CUDA-сборки не взлетели (нет CUDA Toolkit нужной версии?) — пробую CPU-колёса...")
    if install_cpu_prebuilt():
        say("    Подсказка: на твоей видеокарте ответы будут быстрее, если поставить")
        say("    CUDA Toolkit (тогда перезапусти setup_llama.bat): https://developer.nvidia.com/cuda-downloads")
        return True
    say("[Capri] Последняя попытка: сборка из исходников (нужен компилятор C++)...")
    if run([sys.executable, "-m", "pip", "install", "llama-cpp-python"]):
        if _import_check("исходники"):
            return True
    else:
        say("[!] Сборка из исходников не удалась — на Windows нет компилятора C/C++")
        say("    (ошибка 'nmake: no such file or directory' — это ровно оно).")
    say("")
    say("[!] Не удалось поставить llama-cpp-python ни одной готовой сборкой. Варианты:")
    say("    1) Проще всего — вернуться на Ollama: setup_ai.bat (всё в одном, без wheels).")
    say("    2) Поставить CUDA Toolkit и перезапустить setup_llama.bat:")
    say("       https://developer.nvidia.com/cuda-downloads")
    say("    3) Visual Studio Build Tools (галочка Desktop development with C++):")
    say("       https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/")
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
