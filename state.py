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
# How much of each card can actually be handed out, learned by watching.
# A card's sticker VRAM is not its budget: Windows keeps a reserve for the
# desktop and gives a process less than the raw total, so a load can spill
# with gigabytes apparently free. A model that spilled measures that budget
# - if more could have been placed on the card, it would have been - which
# makes this measurable rather than a guess.
VRAM_LIMITS_PATH = os.path.join(ROOT, "vram_limits.json")

# The measurement is the *peak* of a spill, never a sample taken part-way
# through one. A load fills a card over tens of seconds and starts spilling
# well before it has finished filling, so most of the readings taken during
# a spill are of a card still being filled rather than of a card that is
# full. Two things keep those readings out:
#
#   - every number learned here is a maximum, so a small reading loses to
#     the peak of the same load rather than replacing it;
#   - a reading only counts once the card's whole footprint has been flat
#     for VRAM_SETTLE_SECONDS. Dedicated usage alone will not do: a card
#     that has run out goes flat while the model keeps loading into system
#     RAM, so what is watched is what is on the card plus what it pushed
#     off it.
VRAM_SETTLE_SECONDS = 8.0
VRAM_SETTLE_SLACK = 64 * 1024 ** 2

# The lowest ceiling worth believing, as a reserve the card could plausibly
# be holding back and as a share of its sticker VRAM. A ceiling under that
# was not measured on a full card: a 16 GB card that appears to stop handing
# out at 7 GB was read while a load was still filling it. Entries like that
# are in the file this code inherited, so the rule applies on the way in as
# well as on the way out.
#
# Both forms are needed because they fail at opposite ends. The reserve is
# what actually happens - this machine idles at up to 2.4 GB of dedicated
# VRAM before anything is loaded, and Windows holds back more on top of
# that, so 5 GB is already generous - but subtracting a fixed 5 GB from a
# 4 GB card believes anything. The share covers the small cards; the
# reserve stops a 16 GB card being written off at 9 GB, which the share
# alone would wave through.
VRAM_MAX_RESERVE = 5 * 1024 ** 3
VRAM_MIN_CEILING = 0.6


def least_credible_ceiling(vram_bytes):
    """The lowest ceiling that could be a real reserve on a card this size."""
    return max(int(vram_bytes) - VRAM_MAX_RESERVE,
               int(int(vram_bytes) * VRAM_MIN_CEILING))

# How long a ceiling is believed without being confirmed again. A ceiling
# has to be re-earned rather than bind forever, because it suppresses the
# evidence that would move it: the estimate, "Use largest context" and the
# suggested -ts all plan within the ceiling, so nothing ever asks the card
# for more and no clean run can ever prove it wrong. Letting it lapse means
# the app probes upwards again every so often, and either learns the same
# number back or learns a better one.
VRAM_CEILING_DAYS = 30

# How often a spill already on record is worth writing down again. A model
# can sit there spilling for half an hour, which at one counter sample
# every couple of seconds is a thousand readings of the same fact: that is
# how "seen" came to read 1064. Re-confirming at most this often keeps the
# ceiling's age meaningful - it is what VRAM_CEILING_DAYS counts from -
# without rewriting the file all afternoon.
VRAM_CONFIRM_SECONDS = 3600.0

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


def find_preset(candidates, backend_name, name):
    """The saved preset with this name for this model and backend, or None."""
    return next((p for p in presets_for(candidates, backend_name)
                 if p.get("name") == name), None)


def save_preset(candidates, preset):
    """Add a preset, or replace the one with the same name and backend in
    place, so the dropdown keeps its order.

    It goes under whichever key the file already uses for this model - the
    same weights can be listed as a tag or a .gguf name - and a new model
    goes under the first handle given. Works on the raw file so entries
    load_presets() would skip are kept as they were.
    """
    raw = read_json(PRESETS_PATH, {})
    models = raw.get("models")
    if not isinstance(models, dict):
        models = {}
    wanted = {preset_key(c) for c in candidates if c}
    key = next((k for k in models if preset_key(k) in wanted), None) \
        or next(c for c in candidates if c)
    items = list(models.get(key) or [])
    for i, old in enumerate(items):
        if isinstance(old, dict) and old.get("name") == preset["name"] \
                and old.get("backend") in (None, "", preset.get("backend")):
            items[i] = preset
            break
    else:
        items.append(preset)
    models[key] = items
    return save_presets(models)


def load_history(limit=None):
    """Past requests, newest first. Never raises: a corrupt file costs the
    history, not the app."""
    rows = [r for r in (read_json(HISTORY_PATH, {}).get("rows") or [])
            if isinstance(r, dict)]
    return rows[:limit] if limit else rows


def save_history(rows):
    return write_json(HISTORY_PATH, {"version": 1, "rows": list(rows)})


def load_vram_limits():
    """{luid: {"vram", "ceiling", "clean", "seen", "at", "ts"}}.

    "seen" counts confirmations of the ceiling rather than readings, and
    "at"/"ts" are when the last one landed.

    Pruned on the way in, and the file rewritten when pruning changed
    anything, so a limit that can no longer be believed stops being quoted
    back at a user who opens the file to see what the app thinks.
    """
    cards = read_json(VRAM_LIMITS_PATH, {}).get("cards")
    cards = cards if isinstance(cards, dict) else {}
    if prune_vram_limits(cards):
        save_vram_limits(cards)
    return cards


def save_vram_limits(cards):
    return write_json(VRAM_LIMITS_PATH, {"version": 1, "cards": cards})


def _confirmed_age(card, now):
    """Seconds since this card's ceiling was last confirmed, or None when
    the entry does not say."""
    stamp = card.get("ts")
    if not isinstance(stamp, (int, float)) or stamp <= 0:
        try:
            stamp = time.mktime(time.strptime(str(card.get("at") or ""),
                                              "%Y-%m-%d %H:%M"))
        except (ValueError, OverflowError):
            return None
    return max(0.0, now - float(stamp))


def prune_vram_limits(cards, now=None):
    """Forget ceilings that cannot be believed. True if anything changed.

    A ceiling goes for one of two reasons: it is too far below the card's
    sticker VRAM to be a real reserve (least_credible_ceiling), or nothing has
    confirmed it for VRAM_CEILING_DAYS - see that constant for why a
    ceiling has to expire. The floor a clean run proved is never dropped,
    because it is a fact about the card rather than an inference, and the
    card's row stays either way so the history is still readable.
    """
    now = time.time() if now is None else now
    changed = False
    for luid, card in list(cards.items()):
        if not isinstance(card, dict):
            del cards[luid]
            changed = True
            continue
        ceiling = int(card.get("ceiling") or 0)
        if not ceiling:
            continue
        vram_bytes = int(card.get("vram") or 0)
        age = _confirmed_age(card, now)
        if (vram_bytes and ceiling < least_credible_ceiling(vram_bytes)) or \
                (age is not None and age > VRAM_CEILING_DAYS * 86400):
            for key in ("ceiling", "seen", "at", "ts"):
                card.pop(key, None)
            changed = True
    return changed


def _settled(watch, luid, footprint):
    """True once this card's footprint has stopped growing.

    `watch` is scratch the caller keeps between polls, keyed by LUID; None
    means take every sample, which is what a test or a one-shot caller
    wants. The comparison is against the reading at the start of the flat
    stretch rather than against the previous poll, so a load creeping up a
    few MB at a time is still seen to be growing.
    """
    if watch is None:
        return True
    now = time.time()
    flat = watch.get(luid)
    if flat is None or footprint > flat["base"] + VRAM_SETTLE_SLACK:
        watch[luid] = {"base": footprint, "since": now}
        return False
    return now - flat["since"] >= VRAM_SETTLE_SECONDS


def note_vram(cards, luid, vram_bytes, used, spilled=0, watch=None):
    """Fold one observation of a card into what is known about its budget.

    `used` is everything on the card; `spilled` is what it pushed into
    system RAM, and zero means a clean sample.

    Both numbers learned here are the *most* the card was ever seen to hand
    out, never the least. Taking the least looks careful and is wrong: a
    load climbs to its limit through every value below it, so the smallest
    reading during a spill is the start of the fill rather than the limit.
    The peak of a spill is the answer, because at the peak the card was as
    full as it was going to get and still would not take the rest.

    Spilling makes that peak a ceiling as well as a floor; a clean run only
    makes it a floor. A spill never lowers the floor - a load that fitted
    yesterday still fitted - which is the other half of the old rule that
    had to go.

    Returns True when anything changed and the file is worth writing.
    """
    if not luid or not used or not vram_bytes:
        return False
    used, spilled, vram_bytes = int(used), int(spilled or 0), int(vram_bytes)
    if not _settled(watch, luid, used + spilled):
        return False
    card = dict(cards.get(luid) or {})
    before = dict(card)
    card["vram"] = vram_bytes
    if spilled:
        ceiling = max(int(card.get("ceiling") or 0), used)
        age = _confirmed_age(card, time.time())
        # A reading from part-way up a load can still get this far - the
        # watch only sees the samples it is handed, and a slow disk can
        # hold one still for a long time - so refuse outright a ceiling too
        # low to be a desktop reserve.
        if ceiling >= least_credible_ceiling(vram_bytes) and (
                ceiling > int(card.get("ceiling") or 0) or age is None
                or age >= VRAM_CONFIRM_SECONDS):
            card["ceiling"] = ceiling
            card["seen"] = int(card.get("seen") or 0) + 1
            card["at"] = time.strftime("%Y-%m-%d %H:%M")
            card["ts"] = int(time.time())
    else:
        card["clean"] = max(int(card.get("clean") or 0), used)
    if card == before:
        return False
    cards[luid] = card
    return True


def vram_budget(cards, luid, vram_bytes):
    """What one card can really hand out, or its sticker VRAM if unknown."""
    card = cards.get(luid) or {}
    vram_bytes = int(vram_bytes)
    ceiling = int(card.get("ceiling") or 0)
    if ceiling < least_credible_ceiling(vram_bytes):
        ceiling = 0        # never credible; see prune_vram_limits
    if not ceiling:
        return vram_bytes
    # Both are amounts the card was seen to hand out, so the larger is the
    # one that has actually been proved: a clean run above the ceiling says
    # the ceiling has moved, and nothing here may read below what already
    # worked.
    return min(vram_bytes, max(ceiling, int(card.get("clean") or 0)))


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
