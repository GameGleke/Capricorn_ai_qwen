"""Управление мышью: клики, перемещение, двойной клик, колесо в точку.

SendInput вместо mouse_event — так клики проходят и в играх/ Chromium-приложениях,
которые игнорируют устаревший API. Координаты нормализованы под виртуальный экран
(несколько мониторов с отрицательными координатами работают корректно).
"""
import ctypes
import logging
import re
import time
from ctypes import wintypes

log = logging.getLogger("capri")
u = ctypes.windll.user32

ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


MOUSEEVENTF_MOVE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0001, 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
MOUSEEVENTF_WHEEL, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x0800, 0x8000, 0x4000

BUTTONS = {"левая": MOUSEEVENTF_LEFTDOWN, "правая": MOUSEEVENTF_RIGHTDOWN, "средняя": MOUSEEVENTF_MIDDLEDOWN}
BUTTON_UP = {MOUSEEVENTF_LEFTDOWN: MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTDOWN: MOUSEEVENTF_RIGHTUP,
             MOUSEEVENTF_MIDDLEDOWN: MOUSEEVENTF_MIDDLEUP}


def _send_mouse(dx, dy, flags, data=0):
    inp = INPUT(type=0, u=_INPUTUNION(mi=MOUSEINPUT(dx, dy, data, flags, 0, 0)))
    arr = (INPUT * 1)(inp)
    return u.SendInput(1, arr, ctypes.sizeof(INPUT))


def virtual_desk():
    """Прямоугольник всех монитов вместе (SM_XVIRTUALSCREEN=76..79)."""
    x = u.GetSystemMetrics(76)
    y = u.GetSystemMetrics(77)
    return x, y, u.GetSystemMetrics(78), u.GetSystemMetrics(79)


def cursor_pos():
    p = wintypes.POINT()
    u.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def move_to(x, y, smooth=True):
    """Переместить курсор в абсолютную точку экрана."""
    vx, vy, vw, vh = virtual_desk()
    if smooth:
        cx, cy = cursor_pos()
        steps = max(1, min(25, int((abs(cx - x) + abs(cy - y)) / 60)))
        for i in range(1, steps + 1):
            ix = int(cx + (x - cx) * i / steps)
            iy = int(cy + (y - cy) * i / steps)
            nx = int((ix - vx) * 65535 / max(1, vw - 1))
            ny = int((iy - vy) * 65535 / max(1, vh - 1))
            _send_mouse(nx, ny, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)
            time.sleep(0.008)
    nx = int((x - vx) * 65535 / max(1, vw - 1))
    ny = int((y - vy) * 65535 / max(1, vh - 1))
    _send_mouse(nx, ny, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)
    return True


def click(button="left", x=None, y=None, double=False, down_only=False, up_only=False):
    """Клик (или зажать/отпустить). Если заданы x,y — сначала переносим курсор туда."""
    bflag = BUTTONS.get(button, MOUSEEVENTF_LEFTDOWN)
    if x is not None and y is not None:
        move_to(int(x), int(y))
        time.sleep(0.05)
    if down_only:
        return bool(_send_mouse(0, 0, bflag))
    if up_only:
        return bool(_send_mouse(0, 0, BUTTON_UP[bflag]))
    ok = _send_mouse(0, 0, bflag)
    time.sleep(0.03)
    ok &= _send_mouse(0, 0, BUTTON_UP[bflag])
    if double:
        time.sleep(0.05)
        ok &= _send_mouse(0, 0, bflag)
        time.sleep(0.03)
        ok &= _send_mouse(0, 0, BUTTON_UP[bflag])
    return bool(ok)


def drag(x1, y1, x2, y2, button="left"):
    """Зажать левую кнопку в точке A, дотащить до B, отпустить — выделять текст, таскать окна."""
    bflag = BUTTONS.get(button, MOUSEEVENTF_LEFTDOWN)
    move_to(x1, y1, smooth=False)
    time.sleep(0.05)
    _send_mouse(0, 0, bflag)
    move_to(x2, y2, smooth=True)
    time.sleep(0.05)
    return bool(_send_mouse(0, 0, BUTTON_UP[bflag]))


def wheel(amount, x=None, y=None):
    """Колесо мыши: amount>0 — вверх, <0 — вниз (в щелчках по 120)."""
    if x is not None and y is not None:
        move_to(int(x), int(y), smooth=False)
        time.sleep(0.03)
    return bool(_send_mouse(0, 0, MOUSEEVENTF_WHEEL, ctypes.c_uint32(int(amount * 120) & 0xFFFFFFFF).value))


# ---------- разбор фраз ----------

BTN_WORDS = {"прав": "right", "права": "right", "правая": "right", "правой": "right",
             "средн": "middle", "средняя": "middle", "колесико": "middle"}


def _center_xy(t):
    """Центр экрана (или верхнего левого монитора при мультимониторе)."""
    vx, vy, vw, vh = virtual_desk()
    if re.search(r"(лев|монитор)", t) and vx < 0:
        # первый монитор начинается в (0,0) — берём его размеры из SM_CXSCREEN
        return int(vw * 0 + u.GetSystemMetrics(0) // 2), int(u.GetSystemMetrics(1) // 2)
    return vx + vw // 2, vy + vh // 2


def parse_click(text):
    """«кликни правой кнопкой», «двойной клик по центру», «кликни на 500 300»,
    «кликни в центр экрана», «поставь курсор в 100 200», «зажми левую кнопку»,
    «отпусти кнопку», «перетащи с 100 100 на 400 400». Возвращает dict или None."""
    t = text.lower().replace("ё", "е")
    if not re.search(r"клик|щелк|нажми кнопку|зажми|отпусти|курсор|перетащ|протащ|drag", t):
        return None
    res = {"button": "left", "double": False, "x": None, "y": None,
           "down": False, "up": False, "move": False, "drag": False}
    if re.search(r"\b(прав\w*|средн\w*)\b.*кноп|кнопк\w*\s+(прав\w*|средн\w*)", t):
        m = re.search(r"(прав\w*|средн\w*)", t)
        res["button"] = BTN_WORDS.get(m.group(1)[:5], "right" if "прав" in m.group(1) else "middle")
    if re.search(r"двойн\w*|дабл|дважды", t):
        res["double"] = True
    if re.search(r"зажм|удержив|не отпуск", t):
        res["down"] = True
    if re.search(r"отпуст|разожм", t):
        res["up"] = True
    m = re.search(r"с (?P<x1>\d+)\s*(?P<y1>\d+)(?:\s*(?:на|в|до))\s*(?P<x2>\d+)\s*(?P<y2>\d+)", t)
    if m and re.search(r"перетащ|протащ|тащи|drag", t):
        res["drag"] = (int(m.group("x1")), int(m.group("y1")), int(m.group("x2")), int(m.group("y2")))
        return res
    nums = re.findall(r"\d+", t)
    if len(nums) >= 2:
        res["x"], res["y"] = int(nums[0]), int(nums[1])
    elif any(w in t for w in ("центр экрана", "в центр", "на середин", "посередине", "по центру")):
        res["x"], res["y"] = _center_xy(t)
    if re.search(r"перевед|перемест|подвед|двигай|поставь|курсор (в|на|к)", t) and not res["down"] and not res["up"]:
        # чистое перемещение без клика — только если слов про клик нет
        if not re.search(r"клик|щелк", t):
            res["move"] = True
    return res


def do_click(spec):
    if spec.get("drag"):
        x1, y1, x2, y2 = spec["drag"]
        ok = drag(x1, y1, x2, y2, spec["button"])
        return f"Перетащила из ({x1}, {y1}) в ({x2}, {y2})." if ok else None
    if spec["move"]:
        if spec["x"] is not None:
            move_to(spec["x"], spec["y"])
            return f"Курсор в точке {spec['x']} {spec['y']}."
        return None
    if spec["down"]:
        click(spec["button"], spec["x"], spec["y"], down_only=True)
        return f"Держу {spec['button']} кнопку."
    if spec["up"]:
        click(spec["button"], spec["x"], spec["y"], up_only=True)
        return "Отпустила."
    ok = click(spec["button"], spec["x"], spec["y"], double=spec["double"])
    where = f" в ({spec['x']}, {spec['y']})" if spec["x"] is not None else ""
    return ("Кликнула" + (" дважды" if spec["double"] else "") +
            (f" {spec['button']} кнопкой" if spec["button"] != "left" else "") + where + "." if ok else None)
