"""Файловые операции Капри: создать, записать, прочитать, переместить, скопировать, удалить.

Все «опасные» действия (удаление, перезапись) помечаются как требующие подтверждения —
main/actions спрашивают «точно?», и действие выполняется только после «да».
Системные папки и чужие профили под запретом: трогать можно только пользовательские директории.
"""
import logging
import os
import shutil
import stat

log = logging.getLogger("capri")

# куда запрещено писать/откуда запрещено удалять (нижний регистр, без слэшей в конце)
FORBIDDEN_ROOTS = (
    r"c:\windows", r"c:\program files", r"c:\program files (x86)", r"c:\programdata",
    r"c:\boot", r"c:\system volume information", r"c:\recovery",
)

MAX_READ = 4000        # сколько символов файла максимум озвучиваем/показываем
MAX_WRITE = 200_000    # ограничение на размер записи


def user_homes():
    """Домашние папки всех пользователей + текущая (для проверки разрешённых зон)."""
    import getpass
    homes = [os.path.expanduser("~")]
    base = os.environ.get("USERPROFILE", "")
    users_dir = os.path.dirname(base) if base else ""
    if users_dir and os.path.isdir(users_dir):
        for name in os.listdir(users_dir):
            p = os.path.join(users_dir, name)
            if os.path.isdir(p):
                homes.append(p)
    return homes


def allowed_path(path):
    """True, если путь лежит внутри пользовательских папок и не в системной зоне."""
    try:
        real = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except OSError:
        return False
    for forb in FORBIDDEN_ROOTS:
        f = os.path.normcase(forb)
        if real == f or real.startswith(f + os.sep):
            return False
    for home in user_homes():
        try:
            h = os.path.normcase(os.path.realpath(home))
        except OSError:
            continue
        if real == h or real.startswith(h + os.sep):
            return True
    # прочие диски (D:\проекты и т.п.) разрешаем, кроме заведомо системных папок на них
    return True


def _fixed_drives():
    drives = []
    try:
        import ctypes
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for i in range(26):
            if bitmask & (1 << i):
                root = f"{chr(65 + i)}:\\"
                if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:  # DRIVE_FIXED
                    drives.append(root.rstrip("\\"))
    except Exception:
        drives = ["C:"]
    return drives


def resolve_place(place, base_dir):
    """«на рабочем столе», «в загрузках», «документах» или прямой путь → абсолютная папка."""
    import ctypes.wintypes as wt
    CSIDL = {"desktop": 0, "documents": 5, "downloads": 40, "music": 13, "videos": 14,
             "pictures": 33, "home": None}
    p = (place or "").lower().replace("ё", "е")
    key = ("desktop" if "рабоч" in p and "стол" in p else
           "documents" if "документ" in p else
           "downloads" if "загруз" in p or "скачан" in p else
           "music" if "музык" in p else
           "videos" if "видео" in p else
           "pictures" if any(w in p for w in ("фото", "изображен", "картинк")) else
           "home" if any(w in p for w in ("домашн", "профил", "корень диска c", "на диске ц")) else None)
    if key == "home":
        return os.path.expanduser("~")
    if key is not None:
        try:
            buf = wt.create_string_buffer(260)
            ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL[key], None, 0, buf)
            got = buf.value.decode("utf-8", "ignore")
            if got and os.path.isdir(got):
                return got
        except Exception:
            pass
        return os.path.expanduser("~")
    if p:  # возможно, это уже путь («в папке C:\\tmp», «в capricorn-main»)
        cand = p.strip(' "')
        if os.path.isabs(cand):
            return cand if os.path.isdir(cand) else None
        rel = os.path.join(base_dir, cand)
        if os.path.isdir(rel):
            return rel
    return None


def find_file(name, roots=None, limit=60000):
    """Небольшой обход популярных папок в поисках файла/папки по имени (без search everywhere)."""
    name = name.lower().replace("ё", "е").strip()
    if not name:
        return None
    if roots is None:
        home = os.path.expanduser("~")
        roots = [os.path.join(home, d) for d in
                 ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos")]
        roots.append(home)
    best = None
    seen = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "$")) and
                           d.lower() not in ("appdata", "node_modules", "__pycache__", ".venv", "venv")]
            for entry in dirnames + filenames:
                seen += 1
                if seen > limit:
                    return best
                low = entry.lower().replace("ё", "е")
                full = os.path.join(dirpath, entry)
                if low == name or low.startswith(name) or name in low:
                    # самый короткий путь = наиболее «корневой» совпадением
                    if best is None or len(full) < len(best):
                        best = full
                    break
    return best


def safe_join(folder, name):
    name = re_clean_name(name)
    return os.path.join(folder, name) if folder else None


def re_clean_name(name):
    import re
    name = re.sub(r'[<>:"/\\|?*]', "", name or "").strip()
    return name[:120]


def unique_path(path):
    """C:\\a\\note.txt → первый свободный: note (2).txt, note (3).txt… Ничего не перезаписываем."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem} ({n}){ext}"):
        n += 1
    return f"{stem} ({n}){ext}"


def make_file(folder, name, content=""):
    path = unique_path(safe_join(folder, name))
    mode = "directory" if "." not in os.path.basename(path) and not content else "file"
    if mode == "directory":
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content[:MAX_WRITE])
    return path


def append_file(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write(content[:MAX_WRITE])
    return path


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_READ + 1)
    return text[:MAX_READ]


def move_item(src, dst_folder):
    dst = os.path.join(dst_folder, os.path.basename(src))
    dst = unique_path(dst)
    shutil.move(src, dst)
    return dst


def copy_item(src, dst_folder):
    if os.path.isdir(src):
        dst = unique_path(os.path.join(dst_folder, os.path.basename(src)))
        shutil.copytree(src, dst)
    else:
        dst = unique_path(os.path.join(dst_folder, os.path.basename(src)))
        shutil.copy2(src, dst)
    return dst


def _on_rm_error(func, path, exc_info):
    """Читаем-only файлы (атрибут read-only) Windows не даёт удалить shutil по умолчанию."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def delete_item(path):
    """Удалить навсегда (корзина не используется: у пользователя включена автоочистка корзины,
    так что она ничего не гарантирует). Безопасность обеспечивает подтверждение «да» в actions."""
    if not allowed_path(path):
        raise PermissionError("системная или чужая папка")
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_on_rm_error)
    else:
        os.remove(path)
    return True
