"""Franz - real-time vision-action loop desktop controller."""

import asyncio
import base64
import ctypes
import ctypes.wintypes as W
import http.client
import json
import logging
import re
import struct
import time
import urllib.parse
import webbrowser
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

HERE: Final[Path] = Path(__file__).resolve().parent
CONFIG_PATH: Final[Path] = HERE / "config.py"
PANEL_HTML: Final[Path] = HERE / "panel.html"
NORM: Final[int] = 1000
SRCCOPY: Final[int] = 0x00CC0020
CAPTUREBLT: Final[int] = 0x40000000
HALFTONE: Final[int] = 4
LDN: Final[int] = 0x0002
LUP: Final[int] = 0x0004
RDN: Final[int] = 0x0008
RUP: Final[int] = 0x0010

log = logging.getLogger("franz")


def _load_cfg() -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", str(CONFIG_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_C: Any = _load_cfg()


def cfg(name: str, default: Any = None) -> Any:
    return getattr(_C, name, default)


def clamp(v: int, lo: int = 0, hi: int = NORM) -> int:
    return max(lo, min(hi, v))


def safe_int(v: Any) -> int:
    try:
        return clamp(int(float(v)))
    except (ValueError, TypeError):
        return 0


def setup_logging(run_dir: Path) -> None:
    level = getattr(logging, str(cfg("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter("[%(name)s][%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if cfg("LOG_TO_FILE", True):
        fh = logging.FileHandler(run_dir / "main.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def make_run_dir() -> Path:
    base = HERE / str(cfg("RUNS_DIR", "runs"))
    base.mkdir(exist_ok=True)
    n = sum(1 for d in base.iterdir() if d.is_dir() and d.name.startswith("run_"))
    rd = base / f"run_{n + 1:04d}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


@dataclass
class Ghost:
    x1: int
    y1: int
    x2: int
    y2: int
    turn: int
    image_b64: str
    label: str = ""


@dataclass
class State:
    phase: str = "init"
    error: str | None = None
    turn: int = 0
    run_dir: Path | None = None
    annotated_b64: str = ""
    raw_b64: str = ""
    raw_seq: int = 0
    vlm_json: str = ""
    observation: str = ""
    bboxes: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    ghosts: list[dict[str, Any]] = field(default_factory=list)
    parse_level: int = 0
    msg_id: int = 0
    pending_seq: int = 0
    annotated_seq: int = -1
    annotated_event: asyncio.Event = field(default_factory=asyncio.Event)
    next_vlm: str | None = None
    next_event: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


S: State
STOP: asyncio.Event
GHOST_RING: deque[Ghost] = deque()


def set_phase(p: str, err: str | None = None) -> None:
    S.phase, S.error = p, err
    log.info("phase=%s err=%s", p, err)


try:
    ctypes.WinDLL("shcore", use_last_error=True).SetProcessDpiAwareness(2)
except Exception:
    pass

_u32 = ctypes.WinDLL("user32", use_last_error=True)
_g32 = ctypes.WinDLL("gdi32", use_last_error=True)


def _s(dll: Any, nm: str, at: list[Any], rt: Any) -> None:
    f = getattr(dll, nm); f.argtypes = at; f.restype = rt


_s(_u32, "GetDC", [W.HWND], W.HDC)
_s(_u32, "ReleaseDC", [W.HWND, W.HDC], ctypes.c_int)
_s(_u32, "GetSystemMetrics", [ctypes.c_int], ctypes.c_int)
_s(_g32, "CreateCompatibleDC", [W.HDC], W.HDC)
_s(_g32, "CreateDIBSection", [W.HDC, ctypes.c_void_p, W.UINT, ctypes.POINTER(ctypes.c_void_p), W.HANDLE, W.DWORD], W.HBITMAP)
_s(_g32, "SelectObject", [W.HDC, W.HGDIOBJ], W.HGDIOBJ)
_s(_g32, "BitBlt", [W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.HDC, ctypes.c_int, ctypes.c_int, W.DWORD], W.BOOL)
_s(_g32, "StretchBlt", [W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.DWORD], W.BOOL)
_s(_g32, "SetStretchBltMode", [W.HDC, ctypes.c_int], ctypes.c_int)
_s(_g32, "SetBrushOrgEx", [W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p], W.BOOL)
_s(_g32, "DeleteObject", [W.HGDIOBJ], W.BOOL)
_s(_g32, "DeleteDC", [W.HDC], W.BOOL)
_s(_u32, "SetCursorPos", [ctypes.c_int, ctypes.c_int], W.BOOL)
_s(_u32, "mouse_event", [W.DWORD, W.DWORD, W.DWORD, W.DWORD, ctypes.c_void_p], None)


class _BIH(ctypes.Structure):
    _fields_ = [("biSize", W.DWORD), ("biWidth", W.LONG), ("biHeight", W.LONG), ("biPlanes", W.WORD), ("biBitCount", W.WORD), ("biCompression", W.DWORD), ("biSizeImage", W.DWORD), ("biXPelsPerMeter", W.LONG), ("biYPelsPerMeter", W.LONG), ("biClrUsed", W.DWORD), ("biClrImportant", W.DWORD)]


class _BMI(ctypes.Structure):
    _fields_ = [("bmiHeader", _BIH), ("bmiColors", W.DWORD * 3)]


def _bmi(w: int, h: int) -> _BMI:
    b = _BMI()
    hd = b.bmiHeader
    hd.biSize, hd.biWidth, hd.biHeight, hd.biPlanes, hd.biBitCount, hd.biCompression = ctypes.sizeof(_BIH), w, -h, 1, 32, 0
    return b


def _screen() -> tuple[int, int]:
    w, h = int(_u32.GetSystemMetrics(0)), int(_u32.GetSystemMetrics(1))
    return (w, h) if w > 0 and h > 0 else (1920, 1080)


def _crop_px(bw: int, bh: int) -> tuple[int, int, int, int]:
    c = cfg("CAPTURE_CROP", {"x1": 0, "y1": 0, "x2": NORM, "y2": NORM})
    if not isinstance(c, dict):
        return 0, 0, bw, bh
    x1, y1 = clamp(int(c.get("x1", 0))), clamp(int(c.get("y1", 0)))
    x2, y2 = clamp(int(c.get("x2", NORM))), clamp(int(c.get("y2", NORM)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    ne = lambda v, s: (v * s + NORM // 2) // NORM
    return clamp(ne(x1, bw), 0, bw), clamp(ne(y1, bh), 0, bh), clamp(ne(x2, bw), 0, bw), clamp(ne(y2, bh), 0, bh)


def _n2s(nx: int, ny: int) -> tuple[int, int]:
    sw, sh = _screen()
    x1, y1, x2, y2 = _crop_px(sw, sh)
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    npt = lambda v, span: 0 if span <= 1 else (clamp(v) * (span - 1) + NORM // 2) // NORM
    return x1 + npt(nx, cw), y1 + npt(ny, ch)


def _dib(dc: Any, w: int, h: int) -> tuple[Any, int]:
    bits = ctypes.c_void_p()
    hbmp = _g32.CreateDIBSection(dc, ctypes.byref(_bmi(w, h)), 0, ctypes.byref(bits), None, 0)
    return (hbmp, int(bits.value)) if hbmp and bits.value else (None, 0)


def _capture_full() -> tuple[bytes, int, int] | None:
    sw, sh = _screen()
    sdc = _u32.GetDC(0)
    if not sdc:
        return None
    mdc = _g32.CreateCompatibleDC(sdc)
    if not mdc:
        _u32.ReleaseDC(0, sdc); return None
    hb, bits = _dib(sdc, sw, sh)
    if not hb:
        _g32.DeleteDC(mdc); _u32.ReleaseDC(0, sdc); return None
    old = _g32.SelectObject(mdc, hb)
    _g32.BitBlt(mdc, 0, 0, sw, sh, sdc, 0, 0, SRCCOPY | CAPTUREBLT)
    raw = bytes((ctypes.c_ubyte * (sw * sh * 4)).from_address(bits))
    _g32.SelectObject(mdc, old); _g32.DeleteObject(hb); _g32.DeleteDC(mdc); _u32.ReleaseDC(0, sdc)
    return raw, sw, sh


def _crop_bgra(bgra: bytes, sw: int, sh: int, x1: int, y1: int, x2: int, y2: int) -> tuple[bytes, int, int]:
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return bgra, sw, sh
    if x1 == 0 and y1 == 0 and cw == sw and ch == sh:
        return bgra, sw, sh
    src, out, ss, ds = memoryview(bgra), bytearray(cw * ch * 4), sw * 4, cw * 4
    for y in range(ch):
        so, do = (y1 + y) * ss + x1 * 4, y * ds
        out[do:do + ds] = src[so:so + ds]
    return bytes(out), cw, ch


def _stretch(bgra: bytes, sw: int, sh: int, dw: int, dh: int) -> bytes | None:
    sdc = _u32.GetDC(0)
    if not sdc:
        return None
    sdc2, ddc = _g32.CreateCompatibleDC(sdc), _g32.CreateCompatibleDC(sdc)
    if not sdc2 or not ddc:
        if sdc2: _g32.DeleteDC(sdc2)
        if ddc: _g32.DeleteDC(ddc)
        _u32.ReleaseDC(0, sdc); return None
    sb, sbi = _dib(sdc, sw, sh)
    if not sb:
        _g32.DeleteDC(sdc2); _g32.DeleteDC(ddc); _u32.ReleaseDC(0, sdc); return None
    ctypes.memmove(sbi, bgra, sw * sh * 4)
    os = _g32.SelectObject(sdc2, sb)
    db, dbi = _dib(sdc, dw, dh)
    if not db:
        _g32.SelectObject(sdc2, os); _g32.DeleteObject(sb); _g32.DeleteDC(sdc2); _g32.DeleteDC(ddc); _u32.ReleaseDC(0, sdc); return None
    od = _g32.SelectObject(ddc, db)
    _g32.SetStretchBltMode(ddc, HALFTONE); _g32.SetBrushOrgEx(ddc, 0, 0, None)
    _g32.StretchBlt(ddc, 0, 0, dw, dh, sdc2, 0, 0, sw, sh, SRCCOPY)
    result = bytes((ctypes.c_ubyte * (dw * dh * 4)).from_address(dbi))
    _g32.SelectObject(ddc, od); _g32.SelectObject(sdc2, os)
    _g32.DeleteObject(db); _g32.DeleteObject(sb); _g32.DeleteDC(ddc); _g32.DeleteDC(sdc2); _u32.ReleaseDC(0, sdc)
    return result


def _to_png(bgra: bytes, w: int, h: int) -> bytes:
    stride, src, rows = w * 4, memoryview(bgra), bytearray()
    for y in range(h):
        rows.append(0)
        row = src[y * stride:(y + 1) * stride]
        for i in range(0, len(row), 4):
            rows.extend((row[i + 2], row[i + 1], row[i], 255))
    def ck(t: bytes, b: bytes) -> bytes:
        c = t + b
        return struct.pack(">I", len(b)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + ck(b"IDAT", zlib.compress(bytes(rows), 6)) + ck(b"IEND", b"")


def _bbox_crop_b64(bgra: bytes, img_w: int, img_h: int, bb: dict[str, Any]) -> str:
    px1 = clamp(bb["x1"] * img_w // NORM, 0, img_w)
    py1 = clamp(bb["y1"] * img_h // NORM, 0, img_h)
    px2 = clamp(bb["x2"] * img_w // NORM, 0, img_w)
    py2 = clamp(bb["y2"] * img_h // NORM, 0, img_h)
    cw, ch = px2 - px1, py2 - py1
    if cw <= 0 or ch <= 0:
        return ""
    cropped, cw2, ch2 = _crop_bgra(bgra, img_w, img_h, px1, py1, px2, py2)
    return base64.b64encode(_to_png(cropped, cw2, ch2)).decode("ascii")


def capture() -> tuple[str, int, int, bytes, int, int]:
    d = float(cfg("CAPTURE_DELAY", 0.0))
    if d > 0:
        time.sleep(d)
    cap = _capture_full()
    if not cap:
        return "", 0, 0, b"", 0, 0
    bgra, w, h = cap
    cr = cfg("CAPTURE_CROP")
    if isinstance(cr, dict) and all(k in cr for k in ("x1", "y1", "x2", "y2")):
        bgra, w, h = _crop_bgra(bgra, w, h, *_crop_px(w, h))
    raw_bgra = bgra
    crop_w, crop_h = w, h
    ow, oh = int(cfg("CAPTURE_WIDTH", 0)), int(cfg("CAPTURE_HEIGHT", 0))
    dw, dh = 0, 0
    if ow > 0 and oh > 0:
        dw, dh = ow, oh
    else:
        p = int(cfg("CAPTURE_SCALE_PERCENT", 100))
        if p > 0 and p != 100:
            dw, dh = max(1, (w * p + 50) // 100), max(1, (h * p + 50) // 100)
    if dw > 0 and dh > 0 and (w, h) != (dw, dh):
        s = _stretch(bgra, w, h, dw, dh)
        if s:
            bgra, w, h = s, dw, dh
    b64 = base64.b64encode(_to_png(bgra, w, h)).decode("ascii")
    log.info("capture %dx%d crop=%dx%d b64=%d", w, h, crop_w, crop_h, len(b64))
    return b64, w, h, raw_bgra, crop_w, crop_h


def _build_ghosts(bboxes: list[dict[str, Any]], raw_bgra: bytes, img_w: int, img_h: int, turn: int) -> None:
    max_ghosts = int(cfg("GHOST_MAX", 12))
    for bb in bboxes:
        crop_b64 = _bbox_crop_b64(raw_bgra, img_w, img_h, bb)
        if not crop_b64:
            continue
        GHOST_RING.append(Ghost(
            x1=bb["x1"], y1=bb["y1"], x2=bb["x2"], y2=bb["y2"],
            turn=turn, image_b64=crop_b64, label=bb.get("label", ""),
        ))
    while len(GHOST_RING) > max_ghosts:
        GHOST_RING.popleft()


def _ghosts_for_state(current_turn: int) -> list[dict[str, Any]]:
    max_age = int(cfg("GHOST_MAX_AGE", 6))
    out: list[dict[str, Any]] = []
    for g in GHOST_RING:
        age = current_turn - g.turn
        if age > max_age:
            continue
        out.append({
            "x1": g.x1, "y1": g.y1, "x2": g.x2, "y2": g.y2,
            "turn": g.turn, "age": age, "image_b64": g.image_b64, "label": g.label,
        })
    return out


_RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_RE_BRACES = re.compile(r"\{.*\}", re.S)
_RE_OBS = re.compile(
    r"""(?:observation|phenomenology)\s*[:=]\s*["']?(.*?)["']?\s*(?=[,}\n]|$)""",
    re.I | re.S,
)
_RE_BBOX = re.compile(
    r"(?:bbox|box|B\d)\s*[:$$({]\s*"
    r"(\d{1,4})\s*[,;]\s*(\d{1,4})\s*[,;]\s*(\d{1,4})\s*[,;]\s*(\d{1,4})"
    r"\s*[$$)}]?",
    re.I,
)
_RE_ACTION = re.compile(
    r"(click|right_click|double_click|drag|move)\s*[(]\s*"
    r"(\d{1,4})\s*[,;]\s*(\d{1,4})"
    r"(?:\s*[,;\->]+\s*(\d{1,4})\s*[,;]\s*(\d{1,4}))?"
    r"\s*[)]",
    re.I,
)
_RE_UCI = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.I)
_RE_COORDS = re.compile(
    r"(?:at|position|pos|to|from)?\s*[(]?\s*"
    r"(\d{2,4})\s*[,;]\s*(\d{2,4})"
    r"\s*[)]?",
    re.I,
)


def _uci_drag(uci: str) -> dict[str, Any] | None:
    uci = uci.strip().lower()
    if len(uci) < 4 or uci[0] not in "abcdefgh" or uci[2] not in "abcdefgh":
        return None
    try:
        fr, tr = int(uci[1]), int(uci[3])
    except ValueError:
        return None
    if not (1 <= fr <= 8 and 1 <= tr <= 8):
        return None
    def sq(c: str, r: int) -> tuple[int, int]:
        return clamp((ord(c) - 97) * 125 + 62), clamp((8 - r) * 125 + 62)
    x1, y1 = sq(uci[0], fr)
    x2, y2 = sq(uci[2], tr)
    return {"name": "drag", "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _try_json(raw: str) -> dict[str, Any] | None:
    try:
        o = json.loads(raw)
        if isinstance(o, dict): return o
    except (json.JSONDecodeError, ValueError):
        pass
    if (m := _RE_JSON_BLOCK.search(raw)):
        try:
            o = json.loads(m.group(1))
            if isinstance(o, dict): return o
        except (json.JSONDecodeError, ValueError):
            pass
    if (m := _RE_BRACES.search(raw)):
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict): return o
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _fix_json(raw: str) -> dict[str, Any] | None:
    m = _RE_BRACES.search(raw)
    if not m: return None
    s = m.group(0).replace("'", '"')
    s = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_bboxes(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in raw:
        if isinstance(b, dict) and all(k in b for k in ("x1", "y1", "x2", "y2")):
            e: dict[str, Any] = {"x1": safe_int(b["x1"]), "y1": safe_int(b["y1"]), "x2": safe_int(b["x2"]), "y2": safe_int(b["y2"])}
            if "label" in b: e["label"] = str(b["label"])
            out.append(e)
    return out


def _extract_actions(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in raw:
        if not isinstance(a, dict) or "name" not in a: continue
        nm = str(a["name"]).lower()
        if nm == "chess_move":
            d = _uci_drag(str(a.get("uci", "")))
            if d: out.append(d)
            continue
        if "x1" not in a or "y1" not in a: continue
        e: dict[str, Any] = {"name": nm, "x1": safe_int(a["x1"]), "y1": safe_int(a["y1"])}
        if "x2" in a and "y2" in a:
            e["x2"] = safe_int(a["x2"]); e["y2"] = safe_int(a["y2"])
        out.append(e)
    return out


def _regex_extract(raw: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    custom = cfg("PARSE_CUSTOM_REGEX")
    if isinstance(custom, dict):
        re_obs = re.compile(custom["observation"], re.I | re.S) if "observation" in custom else _RE_OBS
        re_bbox = re.compile(custom["bbox"], re.I) if "bbox" in custom else _RE_BBOX
        re_act = re.compile(custom["action"], re.I) if "action" in custom else _RE_ACTION
    else:
        re_obs, re_bbox, re_act = _RE_OBS, _RE_BBOX, _RE_ACTION
    obs = ""
    if (m := re_obs.search(raw)): obs = m.group(1).strip()
    bboxes = [{"x1": safe_int(m.group(1)), "y1": safe_int(m.group(2)), "x2": safe_int(m.group(3)), "y2": safe_int(m.group(4))} for m in re_bbox.finditer(raw)]
    actions: list[dict[str, Any]] = []
    for m in re_act.finditer(raw):
        a: dict[str, Any] = {"name": m.group(1).lower(), "x1": safe_int(m.group(2)), "y1": safe_int(m.group(3))}
        if m.group(4) and m.group(5):
            a["x2"] = safe_int(m.group(4)); a["y2"] = safe_int(m.group(5))
        actions.append(a)
    if not actions:
        for m in _RE_UCI.finditer(raw):
            d = _uci_drag(m.group(1))
            if d: actions.append(d); break
    return obs, bboxes, actions


def parse_vlm(raw: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], int]:
    raw = raw.strip()
    if not raw: return "", [], [], 5
    obj = _try_json(raw)
    lv = 0
    if obj is None:
        obj = _fix_json(raw)
        lv = 2
    if obj is not None:
        narr = str(obj.get("observation", "") or obj.get("phenomenology", ""))
        bb = _extract_bboxes(obj.get("bboxes", []) if isinstance(obj.get("bboxes"), list) else [])
        ac = _extract_actions(obj.get("actions", []) if isinstance(obj.get("actions"), list) else [])
        log.info("parse lv=%d narr=%d bb=%d ac=%d", lv, len(narr), len(bb), len(ac))
        return narr, bb, ac, lv
    narr, bb, ac = _regex_extract(raw)
    if narr or bb or ac:
        log.warning("parse lv=3 narr=%d bb=%d ac=%d", len(narr), len(bb), len(ac))
        return narr, bb, ac, 3
    actions = [{"name": "click", "x1": safe_int(m.group(1)), "y1": safe_int(m.group(2))} for m in _RE_COORDS.finditer(raw)][:4]
    if actions:
        log.warning("parse lv=4 ac=%d", len(actions))
        return raw[:500], [], actions, 4
    log.error("parse lv=5 len=%d", len(raw))
    return raw[:500], [], [], 5


def _action_echo(actions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for a in actions:
        nm = a.get("name", "?")
        if "x2" in a and "y2" in a:
            parts.append(f"{nm}({a['x1']},{a['y1']}->{a['x2']},{a['y2']})")
        else:
            parts.append(f"{nm}({a.get('x1',0)},{a.get('y1',0)})")
    return ", ".join(parts)


def _jl(path: Path, obj: dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    except Exception as e:
        log.warning("jsonl: %s", e)


def save_turn(rd: Path, turn: int, obs: str, bb: list[Any], ac: list[Any], raw_b64: str) -> None:
    if str(cfg("LOG_LAYOUT", "flat")).lower() == "flat":
        nm = f"turn_{turn:04d}_raw.png"
        if raw_b64:
            try: (rd / nm).write_bytes(base64.b64decode(raw_b64))
            except Exception as e: log.warning("save raw: %s", e)
        _jl(rd / "turns.jsonl", {"turn": turn, "stage": "raw", "observation": obs, "bboxes": bb, "actions": ac, "raw_png": nm})
    else:
        td = rd / f"turn_{turn:04d}"
        td.mkdir(exist_ok=True)
        (td / "vlm.json").write_text(json.dumps({"turn": turn, "observation": obs, "bboxes": bb, "actions": ac}, ensure_ascii=False, indent=2), encoding="utf-8")
        if raw_b64:
            try: (td / "raw.png").write_bytes(base64.b64decode(raw_b64))
            except Exception as e: log.warning("save raw: %s", e)


def save_ann(rd: Path, turn: int, ab64: str) -> None:
    if str(cfg("LOG_LAYOUT", "flat")).lower() == "flat":
        nm = f"turn_{turn:04d}_ann.png"
        try: (rd / nm).write_bytes(base64.b64decode(ab64))
        except Exception as e: log.warning("save ann: %s", e)
        _jl(rd / "turns.jsonl", {"turn": turn, "stage": "annotated", "ann_png": nm})
    else:
        td = rd / f"turn_{turn:04d}"
        td.mkdir(exist_ok=True)
        try: (td / "annotated.png").write_bytes(base64.b64decode(ab64))
        except Exception as e: log.warning("save ann: %s", e)


def _mto(x: int, y: int) -> None:
    _u32.SetCursorPos(x, y)


def _mev(f: int) -> None:
    _u32.mouse_event(f, 0, 0, 0, 0)


def execute(actions: list[dict[str, Any]]) -> None:
    if not cfg("PHYSICAL_EXECUTION", True):
        log.info("exec skip %d", len(actions)); return
    ad = float(cfg("ACTION_DELAY_SECONDS", 0.05))
    ds = max(1, int(cfg("DRAG_DURATION_STEPS", 20)))
    dd = float(cfg("DRAG_STEP_DELAY", 0.01))
    for a in actions:
        nm = a.get("name", "")
        x1, y1 = _n2s(int(a.get("x1", 0)), int(a.get("y1", 0)))
        x2, y2 = _n2s(int(a.get("x2", a.get("x1", 0))), int(a.get("y2", a.get("y1", 0))))
        log.info("exec %s (%d,%d)->(%d,%d)", nm, x1, y1, x2, y2)
        match nm:
            case "move":
                _mto(x1, y1)
            case "click":
                _mto(x1, y1); time.sleep(0.03); _mev(LDN); time.sleep(0.03); _mev(LUP)
            case "right_click":
                _mto(x1, y1); time.sleep(0.03); _mev(RDN); time.sleep(0.03); _mev(RUP)
            case "double_click":
                _mto(x1, y1); time.sleep(0.03); _mev(LDN); time.sleep(0.03); _mev(LUP); time.sleep(0.06); _mev(LDN); time.sleep(0.03); _mev(LUP)
            case "drag":
                _mto(x1, y1); time.sleep(0.03); _mev(LDN); time.sleep(0.03)
                for i in range(1, ds + 1):
                    _mto(x1 + (x2 - x1) * i // ds, y1 + (y2 - y1) * i // ds); time.sleep(dd)
                time.sleep(0.03); _mev(LUP)
            case _:
                log.warning("unknown action %r", nm)
        time.sleep(ad)


def call_vlm(obs: str, ab64: str) -> tuple[str, dict[str, Any], str | None]:
    url = str(cfg("API_URL", ""))
    u = urllib.parse.urlparse(url)
    host, port, path = u.hostname or "127.0.0.1", u.port or 80, u.path or "/v1/chat/completions"
    t = float(cfg("VLM_HTTP_TIMEOUT_SECONDS", 120) or 120)
    timeout = max(30.0, t)
    body = json.dumps({
        "model": str(cfg("MODEL", "")),
        "temperature": float(cfg("TEMPERATURE", 0.7)),
        "top_p": float(cfg("TOP_P", 0.9)),
        "max_tokens": int(cfg("MAX_TOKENS", 1000)),
        "messages": [
            {"role": "system", "content": str(cfg("SYSTEM_PROMPT", ""))},
            {"role": "user", "content": [
                {"type": "text", "text": obs or "(no prior observation)"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ab64}"}},
            ]},
        ],
    }).encode("utf-8")
    log.info("vlm POST %s:%d%s obs=%d img=%d", host, port, path, len(obs), len(ab64))
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json", "Accept": "application/json", "Connection": "close"})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        if not 200 <= resp.status < 300:
            return "", {}, f"HTTP {resp.status}"
        obj = json.loads(data.decode("utf-8", "replace"))
        return cast(str, obj["choices"][0]["message"]["content"]), cast(dict[str, Any], obj.get("usage") or {}), None
    except Exception as e:
        log.error("vlm: %s", e)
        return "", {}, str(e)


async def engine_loop(rd: Path) -> None:
    S.run_dir = rd
    bt, bv = bool(cfg("BOOT_ENABLED", True)), str(cfg("BOOT_VLM_OUTPUT", ""))
    if bt and bv.strip():
        async with S.lock:
            S.next_vlm, _ = bv, S.next_event.set()
        set_phase("boot")
    else:
        set_phase("waiting_inject")
    at = float(cfg("ANNOTATED_TIMEOUT_SECONDS", 15) or 15)
    loop = asyncio.get_running_loop()
    max_lv = int(cfg("PARSE_MAX_LEVEL", 4))
    raw_bgra_buf: bytes = b""
    raw_w, raw_h = 0, 0
    while not STOP.is_set():
        try:
            await asyncio.wait_for(S.next_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        async with S.lock:
            vlm_raw = S.next_vlm or ""
            S.next_vlm = None
            S.next_event.clear()
        if not vlm_raw.strip():
            continue
        S.turn += 1
        turn = S.turn
        log.info("=== TURN %d ===", turn)
        set_phase("running")
        obs, bb, ac, lv = parse_vlm(vlm_raw)
        if lv > max_lv:
            log.warning("parse lv=%d > max=%d, dropping actions", lv, max_lv)
            ac = []
        prefix = f"[TURN {turn}]"
        if ac:
            prefix += f" [EXECUTED: {_action_echo(ac)}]"
        if lv >= 3:
            prefix += f" [FORMAT ERROR lv={lv}: output ONLY valid JSON]"
        obs = f"{prefix}\n{obs}"
        if raw_bgra_buf and bb:
            _build_ghosts(bb, raw_bgra_buf, raw_w, raw_h, turn)
        async with S.lock:
            S.vlm_json, S.observation, S.bboxes, S.actions = vlm_raw, obs, bb, ac
            S.parse_level, S.msg_id = lv, S.msg_id + 1
            S.ghosts = _ghosts_for_state(turn)
        set_phase("executing")
        await loop.run_in_executor(None, execute, ac)
        set_phase("capturing")
        raw_b64, w, h, raw_bgra, crop_w, crop_h = await loop.run_in_executor(None, capture)
        if not raw_b64:
            set_phase("error", "capture failed"); continue
        raw_bgra_buf, raw_w, raw_h = raw_bgra, crop_w, crop_h
        async with S.lock:
            S.raw_b64 = raw_b64
            S.raw_seq += 1
        await loop.run_in_executor(None, save_turn, rd, turn, obs, bb, ac, raw_b64)
        async with S.lock:
            S.pending_seq, S.annotated_seq, S.annotated_b64 = turn, -1, ""
            S.annotated_event.clear()
        set_phase("waiting_annotated")
        got = False
        try:
            await asyncio.wait_for(S.annotated_event.wait(), timeout=at)
            got = True
        except asyncio.TimeoutError:
            log.warning("annotated timeout seq=%d", turn)
        async with S.lock:
            ab64 = S.annotated_b64 if got else raw_b64
        await loop.run_in_executor(None, save_ann, rd, turn, ab64)
        set_phase("calling_vlm")
        txt, usage, err = await loop.run_in_executor(None, call_vlm, obs, ab64)
        if err:
            log.error("vlm err t=%d: %s", turn, err)
            set_phase("vlm_error", err); continue
        log.info("vlm ok t=%d len=%d usage=%s", turn, len(txt), usage)
        async with S.lock:
            S.next_vlm = txt
            S.next_event.set()
        set_phase("running")


class Server:
    def __init__(self, host: str, port: int) -> None:
        self._h, self._p = host, port
        self._srv: asyncio.Server | None = None

    async def start(self) -> None:
        self._srv = await asyncio.start_server(self._conn, self._h, self._p)
        log.info("http://%s:%d", self._h, self._p)

    async def stop(self) -> None:
        if self._srv:
            self._srv.close(); await self._srv.wait_closed()

    async def _conn(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            await self._proc(r, w)
        except (ConnectionResetError, ConnectionAbortedError, asyncio.IncompleteReadError):
            pass
        except OSError as e:
            if getattr(e, "winerror", None) not in (10053, 10054):
                log.warning("conn: %s", e)
        except Exception as e:
            log.warning("conn: %s", e)
        finally:
            try: w.close(); await w.wait_closed()
            except Exception: pass

    async def _proc(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        rl = await asyncio.wait_for(r.readline(), timeout=30)
        if not rl: return
        parts = rl.decode("utf-8", "replace").strip().split(" ")
        if len(parts) < 2: return
        method, path = parts[0], parts[1].split("?", 1)[0]
        hd: dict[str, str] = {}
        while True:
            hl = await asyncio.wait_for(r.readline(), timeout=10)
            if not hl or hl in (b"\r\n", b"\n"): break
            d = hl.decode("utf-8", "replace").strip()
            if ":" in d:
                k, v = d.split(":", 1)
                hd[k.strip().lower()] = v.strip()
        body = b""
        cl = int(hd.get("content-length", "0"))
        if cl > 0:
            body = await asyncio.wait_for(r.readexactly(cl), timeout=60)
        match method:
            case "GET": await self._get(path, w)
            case "POST": await self._post(path, body, w)
            case "OPTIONS": await self._json(w, {})
            case _: await self._err(w, 405)

    async def _get(self, path: str, w: asyncio.StreamWriter) -> None:
        match path:
            case "/" | "/index.html":
                await self._raw(w, 200, "text/html; charset=utf-8", PANEL_HTML.read_bytes())
            case "/config":
                await self._json(w, {"ui": cfg("UI_CONFIG", {}), "capture_width": int(cfg("CAPTURE_WIDTH", 512)), "capture_height": int(cfg("CAPTURE_HEIGHT", 288))})
            case "/state":
                async with S.lock:
                    await self._json(w, {"phase": S.phase, "error": S.error, "turn": S.turn, "msg_id": S.msg_id, "pending_seq": S.pending_seq, "annotated_seq": S.annotated_seq, "raw_seq": S.raw_seq, "bboxes": S.bboxes, "actions": S.actions, "observation": S.observation, "vlm_json": S.vlm_json, "parse_level": S.parse_level, "ghost_count": len(S.ghosts)})
            case "/frame":
                async with S.lock:
                    await self._json(w, {"seq": S.raw_seq, "raw_b64": S.raw_b64})
            case "/ghosts":
                async with S.lock:
                    await self._json(w, {"turn": S.turn, "ghosts": S.ghosts})
            case _:
                await self._err(w, 404)

    async def _post(self, path: str, body: bytes, w: asyncio.StreamWriter) -> None:
        match path:
            case "/annotated":
                try: obj = json.loads(body.decode("utf-8"))
                except Exception: await self._json(w, {"ok": False, "err": "bad json"}, 400); return
                seq, img = obj.get("seq"), obj.get("image_b64", "")
                async with S.lock:
                    exp = S.pending_seq
                if seq != exp:
                    await self._json(w, {"ok": False, "err": f"seq {seq}!={exp}"}, 409); return
                if not isinstance(img, str) or len(img) < 100:
                    await self._json(w, {"ok": False, "err": "img short"}, 400); return
                async with S.lock:
                    S.annotated_b64, S.annotated_seq = img, seq
                    S.annotated_event.set()
                await self._json(w, {"ok": True, "seq": seq})
            case "/inject":
                try: obj = json.loads(body.decode("utf-8"))
                except Exception: await self._json(w, {"ok": False, "err": "bad json"}, 400); return
                txt = obj.get("vlm_text", "")
                if not isinstance(txt, str) or not txt.strip():
                    await self._json(w, {"ok": False, "err": "empty"}, 400); return
                async with S.lock:
                    S.next_vlm = txt
                    S.next_event.set()
                await self._json(w, {"ok": True})
            case _:
                await self._err(w, 404)

    async def _raw(self, w: asyncio.StreamWriter, code: int, ct: str, data: bytes) -> None:
        st = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 409: "Conflict"}.get(code, "OK")
        w.write(f"HTTP/1.1 {code} {st}\r\nContent-Type: {ct}\r\nContent-Length: {len(data)}\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n\r\n".encode() + data)
        await w.drain()

    async def _json(self, w: asyncio.StreamWriter, obj: Any, code: int = 200) -> None:
        await self._raw(w, code, "application/json", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    async def _err(self, w: asyncio.StreamWriter, code: int) -> None:
        await self._json(w, {"error": code}, code)


async def async_main() -> None:
    global S, STOP
    S, STOP = State(), asyncio.Event()
    rd = make_run_dir()
    setup_logging(rd)
    log.info("Franz start rd=%s", rd)
    srv = Server(str(cfg("HOST", "127.0.0.1")), int(cfg("PORT", 1234)))
    await srv.start()
    try: webbrowser.open(f"http://{cfg('HOST', '127.0.0.1')}:{cfg('PORT', 1234)}")
    except Exception: pass
    task = asyncio.create_task(engine_loop(rd))
    try: await STOP.wait()
    except KeyboardInterrupt: STOP.set()
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    await srv.stop()
    log.info("Franz stopped")


def main() -> None:
    try: asyncio.run(async_main())
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()
