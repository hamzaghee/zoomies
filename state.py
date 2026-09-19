r"""
Zoomies - persistent state, paths, and process identity checks.

Everything Zoomies writes lives under %LOCALAPPDATA%\Zoomies, never in the
project folder, so the app can be moved or deleted without losing settings and
so a cluttered project directory never confuses the user.

Two files:
  config.json   user preferences. Survives forever.
  session.json  what is running right now and what we created. Survives a
                crash so the next launch can reconcile and clean up.

The process checks here are deliberately paranoid. Windows recycles PIDs, and
there are already stale .pid files on this machine from weeks ago, so a bare
"does this PID exist" check will eventually point at an unrelated process and
we would happily taskkill it.
"""

import ctypes
import json
import os
import re
import socket
import subprocess
import time
from ctypes import wintypes

APP_NAME = "Zoomies"
TAG_SUFFIX = "-zoomies"          # reserved namespace; see backends.can_remove_tag

CREATE_NO_WINDOW = 0x08000000

_local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
ROOT = os.path.join(_local, APP_NAME)
CACHE_DIR = os.path.join(ROOT, "cache")
SCRIPT_DIR = os.path.join(ROOT, "scripts")
LOG_DIR = os.path.join(ROOT, "logs")

CONFIG_PATH = os.path.join(ROOT, "config.json")
SESSION_PATH = os.path.join(ROOT, "session.json")
# Measured launch settings, one or more per model. Kept beside config.json
# rather than in the project folder because what is fastest depends on the
# machine: the same model on different cards wants different flags.
PRESETS_PATH = os.path.join(ROOT, "presets.json")
# Finished requests, newest first. Kept so the numbers you measured this
# morning are still there tomorrow - the metrics themselves only live as
# long as the process that watched them.
HISTORY_PATH = os.path.join(ROOT, "history.json")

KEEP_FILES = 20                  # how many generated scripts / logs to retain


def ensure_dirs():
    for d in (ROOT, CACHE_DIR, SCRIPT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# atomic json
# --------------------------------------------------------------------------

def read_json(path, default):
    """Never raise. A corrupt state file must not stop the app from starting."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else default
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """Write via a temp file + os.replace so a crash mid-write cannot leave a
    truncated file that we would then fail to parse on the next start."""
    ensure_dirs()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


DEFAULT_CONFIG = {
    "always_on_top": False,
    "unload_on_exit": False,
    "manual_page_map": {},       # model match_key -> docs page title chosen by hand
    "last_backend": "ollama",
    "last_model": "",
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(read_json(CONFIG_PATH, {}))
    return cfg


def save_config(cfg):
    return write_json(CONFIG_PATH, cfg)


DEFAULT_SESSION = {
    "version": 1,
    "ollama": {"created_tags": []},   # tags Zoomies made and must remove again
}


def load_session():
    s = dict(DEFAULT_SESSION)
    raw = read_json(SESSION_PATH, {})
    s.update(raw)
    if not isinstance(s.get("ollama"), dict):
        s["ollama"] = {"created_tags": []}
    s["ollama"].setdefault("created_tags", [])
    return s


def save_session(s):
    return write_json(SESSION_PATH, s)


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

def load_presets():
    """{model id: [preset, ...]}.

    A preset is a name, a settings dict in the same shape settings_dict()
    produces, and an optional note saying where the numbers came from.
    """
    raw = read_json(PRESETS_PATH, {})
    out = {}
    for model_id, items in (raw.get("models") or {}).items():
        good = [p for p in items
                if isinstance(p, dict) and p.get("name")
                and isinstance(p.get("settings"), dict)]
        if good:
            out[model_id] = good
    return out


def save_presets(models):
    return write_json(PRESETS_PATH, {"version": 1, "models": models})


def load_history(limit=None):
    """Past requests, newest first. Never raises: a corrupt file costs the
    history, not the app."""
    rows = [r for r in (read_json(HISTORY_PATH, {}).get("rows") or [])
            if isinstance(r, dict)]
    return rows[:limit] if limit else rows


def save_history(rows):
    return write_json(HISTORY_PATH, {"version": 1, "rows": list(rows)})


def preset_key(text):
    """Normalise a model handle so the same weights match however they arrive.

    The same model is called different things by each backend - an Ollama tag,
    a file path, a Hugging Face name - so presets are matched on a normalised
    form rather than the exact string:

        ornith:35b-q4_K_M                      -> ornith-35b-q4-k-m
        C:\\models\\ornith-35b-Q4_K_M.gguf       -> ornith-35b-q4-k-m
    """
    name = os.path.basename(str(text or "").replace("\\", "/").rstrip("/"))
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


def presets_for(candidates, backend_name):
    """Presets matching any of these handles, on this backend.

    A preset records which backend it was measured on: llama.cpp flags mean
    nothing to Ollama, and applying them silently would be worse than showing
    nothing at all.
    """
    table = load_presets()
    by_key = {}
    for model_id, items in table.items():
        by_key.setdefault(preset_key(model_id), []).extend(items)
    seen = set()
    for candidate in candidates:
        for preset in by_key.get(preset_key(candidate), []):
            name = preset.get("name")
            if name in seen:
                continue
            if preset.get("backend") in (None, "", backend_name):
                seen.add(name)
                yield preset


# --------------------------------------------------------------------------
# process identity
# --------------------------------------------------------------------------

def port_open(host, port, timeout=0.4):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(host, port, timeout=90.0, interval=0.5, cancel=None):
    """Readiness is a TCP connect, never a string match on log output. Log
    wording changes between versions; a listening socket does not."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cancel is not None and cancel.is_set():
            return False
        if port_open(host, port):
            return True
        time.sleep(interval)
    return False


def process_name(pid):
    """Image name for a PID, or empty string if it is gone.

    Uses tasklist because it ships with Windows - psutil would mean a pip
    install, and this app is explicitly standard-library only.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        ).stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
    if not out or out.lower().startswith("info:"):
        return ""
    first = out.splitlines()[0]
    if not first.startswith('"'):
        return ""
    return first.split('","')[0].strip('"')


def alive_and_named(pid, want):
    """True only if the PID exists AND its image name matches.

    The name check is what defends against PID reuse: a recycled PID belonging
    to some unrelated program will not be called powershell.exe.
    """
    if not pid:
        return False
    name = process_name(pid).lower()
    return bool(name) and want.lower() in name


def kill_tree(pid, log_lines=None):
    """taskkill /T /F. The /T is mandatory, not optional polish: a server's
    PowerShell script runs llama.exe as a child, and killing only the parent
    leaves the child holding all the VRAM."""
    if not pid:
        return False
    try:
        res = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(int(pid))],
            capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW,
        )
        if log_lines is not None:
            log_lines.append((res.stdout or res.stderr or "").strip())
        return res.returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def find_processes(image_name):
    """All PIDs for a given image name, e.g. llama-server.exe."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq %s" % image_name, "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        if not line.startswith('"'):
            continue
        parts = line.split('","')
        if len(parts) > 1:
            try:
                pids.append(int(parts[1].strip('"')))
            except ValueError:
                pass
    return pids


# --------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------

def sweep_old_files(directory, keep=KEEP_FILES):
    """Keep the newest files, delete the rest. Called at startup so the scripts
    and logs folders cannot grow without bound."""
    try:
        entries = [
            (os.path.getmtime(os.path.join(directory, n)), os.path.join(directory, n))
            for n in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, n))
        ]
    except OSError:
        return 0
    entries.sort(reverse=True)
    removed = 0
    for _, path in entries[keep:]:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------
# GPUs
# --------------------------------------------------------------------------

# Deliberately measured rather than assumed. A backend's own hardware report
# once claimed a single GPU, and believing it cost this app a wrong VRAM budget:
# this machine has two RX 6800 XTs, not one. The registry also keeps entries
# for cards that are no longer installed (there is a stale RTX 4090 here), so
# a registry sweep alone over-reports. Present devices come from PnP, sizes
# come from the registry, and the two are joined by name.
_GPU_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
_IGNORE_GPU = ("microsoft", "remote display", "basic display", "basic render",
               "virtual", "parsec", "meta ", "citrix")


def _registry_vram():
    """DriverDesc -> bytes, for every adapter the registry knows about."""
    try:
        import winreg
    except ImportError:
        return {}
    sizes = {}
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _GPU_CLASS)
    except OSError:
        return {}
    with root:
        for i in range(256):
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    desc = winreg.QueryValueEx(key, "DriverDesc")[0]
                    size = winreg.QueryValueEx(
                        key, "HardwareInformation.qwMemorySize")[0]
            except OSError:
                continue
            if desc and size:
                sizes[str(desc).strip()] = max(int(size), sizes.get(desc, 0))
    return sizes


def _present_adapters():
    """Names of display adapters actually installed right now."""
    script = ("Get-PnpDevice -Class Display -Status OK -ErrorAction "
              "SilentlyContinue | ForEach-Object { $_.FriendlyName }")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30,
            creationflags=CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


_gpu_cache = None


def detect_gpus(refresh=False):
    """[(name, vram_bytes), ...] for real, present, discrete GPUs.

    Integrated graphics are excluded: they share system RAM, so counting them
    towards a model's VRAM budget would be misleading.
    """
    global _gpu_cache
    if _gpu_cache is not None and not refresh:
        return _gpu_cache

    # DXGI first: it reports dedicated VRAM per adapter, flags software
    # rasterizers, and lists only adapters that actually exist right now -
    # the registry keeps entries for cards that have been removed.
    dxgi = physical_adapters(enumerate_adapters())
    if dxgi:
        _gpu_cache = [(a["name"], a["vram"]) for a in dxgi]
        return _gpu_cache

    sizes = _registry_vram()
    found = []
    for name in _present_adapters():
        low = name.lower()
        if any(bad in low for bad in _IGNORE_GPU):
            continue
        vram = sizes.get(name.strip(), 0)
        if vram >= 1 << 30:          # ignore anything claiming under 1 GB
            found.append((name.strip(), int(vram)))
    _gpu_cache = found
    return found


def total_vram():
    return sum(v for _n, v in detect_gpus())


def largest_vram():
    gpus = detect_gpus()
    return max((v for _n, v in gpus), default=0)


def describe_gpus():
    gpus = detect_gpus()
    if not gpus:
        return "no discrete GPU detected"
    counts = {}
    for name, vram in gpus:
        # Group on whole GB: two identical cards can report VRAM a few KB
        # apart, which would otherwise list them as separate models.
        bucket = (name, round(vram / 1024 ** 3))
        counts.setdefault(bucket, 0)
        counts[bucket] += 1
    parts = ["%s%s (%d GB)" % ("%dx " % n if n > 1 else "", name, gb)
             for (name, gb), n in counts.items()]
    return "%s - %.0f GB total" % (", ".join(parts), total_vram() / 1024 ** 3)


# --------------------------------------------------------------------------
# DXGI adapter enumeration
# --------------------------------------------------------------------------
#
# Ported from the Ollama Monitor, which had already solved this properly.
#
# Windows' performance counters identify a GPU only by LUID, and a LUID is
# handed out per enumeration - the same card gets a different one after a
# reboot, a driver restart or a TDR recovery, and can briefly hold two at
# once. So LUIDs are treated strictly as this-boot handles for joining
# counter rows, and DXGI is used to turn each one into a real adapter name
# plus a vendor/device/subsystem triple that survives reboots.
#
# It also reports DedicatedVideoMemory per adapter, which is a better source
# for the VRAM total than the registry: no stale entries for cards that have
# been removed, and software adapters are flagged rather than guessed at.

DXGI_ADAPTER_FLAG_SOFTWARE = 2

# COM vtable slots. IUnknown occupies 0-2 and IDXGIObject 3-6, which puts
# IDXGIFactory1::EnumAdapters1 at 12 and IDXGIAdapter1::GetDesc1 at 10.
_VT_RELEASE = 2
_VT_ENUM_ADAPTERS1 = 12
_VT_GET_DESC1 = 10


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


_IID_IDXGIFactory1 = _GUID(
    0x770AAE78, 0xF26F, 0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)


def _com_call(iface, slot, restype, argtypes, *args):
    """Invoke a COM method by vtable slot - ctypes has no COM support."""
    vtable = ctypes.cast(
        iface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[slot])
    return fn(iface, *args)


def adapter_key(vendor_id, device_id, subsys_id):
    """Identity that survives reboots."""
    return "%04X:%04X:%08X" % (vendor_id, device_id, subsys_id)


def short_name(description):
    name = re.sub(r"\(R\)|\(TM\)|Corporation", "", description)
    name = re.sub(r"^\s*(NVIDIA|AMD|Intel|Microsoft)\s+", "", name.strip())
    return re.sub(r"\s{2,}", " ", name).strip() or description.strip()


def enumerate_adapters():
    """Every display adapter DXGI knows about, as plain dicts.

    Returns [] rather than raising if DXGI is unavailable - the app has to
    stay useful without GPU numbers.
    """
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return []

    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1),
                               ctypes.byref(factory)) != 0:
        return []

    adapters = []
    try:
        index = 0
        while True:
            iface = ctypes.c_void_p()
            # non-zero is DXGI_ERROR_NOT_FOUND, i.e. the end of the list
            hr = _com_call(factory, _VT_ENUM_ADAPTERS1, ctypes.c_long,
                           [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                           index, ctypes.byref(iface))
            if hr != 0:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                if _com_call(iface, _VT_GET_DESC1, ctypes.c_long,
                             [ctypes.POINTER(_DXGI_ADAPTER_DESC1)],
                             ctypes.byref(desc)) == 0:
                    adapters.append({
                        # two hex halves, matching the counter instance names
                        # exactly so the two sources can be joined on it
                        "luid": "%08X_%08X" % (desc.AdapterLuid.HighPart,
                                               desc.AdapterLuid.LowPart),
                        "name": short_name(desc.Description),
                        "key": adapter_key(desc.VendorId, desc.DeviceId,
                                           desc.SubSysId),
                        "subsys": "%08X" % desc.SubSysId,
                        "vram": int(desc.DedicatedVideoMemory),
                        "is_software": bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                    })
            finally:
                _com_call(iface, _VT_RELEASE, ctypes.c_ulong, [])
            index += 1
    finally:
        _com_call(factory, _VT_RELEASE, ctypes.c_ulong, [])
    return adapters


def physical_adapters(adapters, live_luids=None):
    """One entry per physical card.

    DXGI keeps listing a card under its old LUID after a driver reinstall or
    a TDR recovery, so the same RX 6800 XT can appear twice - which briefly
    made this machine look like it had 48 GB of VRAM instead of 32. The
    vendor/device/subsystem key is identical for both entries, so group on
    it. When live counter data is available, prefer the LUID that is
    actually reporting; otherwise keep the first one DXGI returned.
    """
    live = live_luids or {}
    chosen = {}
    for a in adapters:
        if a["is_software"] or a["vram"] < 1 << 30:
            continue
        held = chosen.get(a["key"])
        if held is None or (held["luid"] not in live and a["luid"] in live):
            chosen[a["key"]] = a
    return list(chosen.values())


def display_label(adapter, among):
    """Disambiguate identical cards by subsystem id, but only when needed."""
    if sum(1 for a in among if a["name"] == adapter["name"]) > 1:
        return "%s (%s)" % (adapter["name"], adapter["subsys"][-4:])
    return adapter["name"]
