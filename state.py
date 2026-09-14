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

import json
import os
import socket
import subprocess
import time

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
    "unsloth_folder": "",
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
    "unsloth": None,                  # dict once a server is running
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
    to some unrelated program will not be called unsloth.exe.
    """
    if not pid:
        return False
    name = process_name(pid).lower()
    return bool(name) and want.lower() in name


def kill_tree(pid, log_lines=None):
    """taskkill /T /F. The /T is mandatory, not optional polish: unsloth.exe
    spawns llama-server.exe as a child, and killing only the parent leaves the
    child holding all the VRAM."""
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
