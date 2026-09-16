r"""
Zoomies - every llama.cpp and Ollama process on the machine, explained.

Loaded only shows what a server's API admits to. Model memory can also sit
in processes no API reports: a `llama cli` chat left open in a terminal, a
server whose launcher has exited, a runner Ollama lost track of. Over a day
of testing those pile up and everything slows down. This lists them all,
says what each one is, and flags the ones nothing accounts for, so they can
be ended without a trip to Task Manager.

Ending is deliberately narrow: only these executables, never Ollama's own
server or tray app, and only after checking the pid still belongs to the
same process that was listed (same start time), because Windows reuses pids.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

import backends
import state

NAMES = ("llama.exe", "llama-server.exe", "ollama.exe", "ollama app.exe")
SHELLS = ("powershell.exe", "pwsh.exe", "cmd.exe", "windowsterminal.exe",
          "conhost.exe", "bash.exe", "wsl.exe", "openconsole.exe")
MODEL_FLAGS = ("-m", "--model", "-hf", "-hfr", "--hf-repo")

# One PowerShell round trip for everything: processes, parents, ports.
_SNAPSHOT_PS = r"""
$names = @('llama.exe', 'llama-server.exe', 'ollama.exe', 'ollama app.exe')
$ports = @{}
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    $ports[[string]$_.OwningProcess] += @([int]$_.LocalPort)
}
$all = @(Get-CimInstance Win32_Process)
$byId = @{}
foreach ($p in $all) { $byId[[string]$p.ProcessId] = $p }
$out = foreach ($p in $all) {
    if ($names -notcontains $p.Name) { continue }
    $parent = $byId[[string]$p.ParentProcessId]
    # A parent that started after its child is a reused pid, not the parent.
    if ($parent -and $p.CreationDate -and $parent.CreationDate -gt $p.CreationDate) { $parent = $null }
    [pscustomobject]@{
        pid        = [int]$p.ProcessId
        name       = [string]$p.Name
        cmd        = [string]$p.CommandLine
        ram        = [int64]$p.WorkingSetSize
        started    = if ($p.CreationDate) { $p.CreationDate.ToString('o') } else { '' }
        ppid       = [int]$p.ParentProcessId
        parent     = if ($parent) { [string]$parent.Name } else { '' }
        parent_cmd = if ($parent) { [string]$parent.CommandLine } else { '' }
        ports      = @($ports[[string]$p.ProcessId] | Sort-Object -Unique)
    }
}
ConvertTo-Json -InputObject @($out) -Depth 3 -Compress
"""


@dataclass
class Proc:
    pid: int
    name: str
    cmd: str
    ram: int
    started: str
    ppid: int
    parent: str
    parent_cmd: str
    ports: list = field(default_factory=list)
    what: str = ""              # plain description shown to the user
    leftover: bool = False      # nothing accounts for it
    protected: bool = False     # never ended from Zoomies
    zoomies_port: int = 0       # a Zoomies llama.cpp server, by port


def snapshot():
    """Raw rows, or [] if PowerShell could not answer."""
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _SNAPSHOT_PS],
            capture_output=True, text=True, timeout=60,
            creationflags=state.CREATE_NO_WINDOW)
        data = json.loads((res.stdout or "").strip() or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    rows = data if isinstance(data, list) else [data]
    return [r for r in rows if isinstance(r, dict) and r.get("pid")]


def _flag(cmd, names):
    """Value of the first of these flags on a command line, or ''."""
    for name in names:
        m = re.search(r'(?:^|\s)%s\s+("[^"]*"|\S+)' % re.escape(name), cmd or "")
        if m:
            return m.group(1).strip('"')
    return ""


def _subcommand(cmd):
    """`serve`, `cli`, `run`... - the first word after the executable."""
    rest = re.sub(r'^\s*("[^"]*"|\S+)\s*', "", cmd or "", count=1)
    word = rest.split(None, 1)[0] if rest.strip() else ""
    return "" if word.startswith("-") else word.lower()


def ollama_blob_names():
    """{"sha256-abc...": "gemma4:12b-it-q4_K_M"} from Ollama's manifests, so a
    runner - which only knows its blob file - can be named."""
    root = os.path.join(backends.ollama_models_dir(), "manifests")
    names = {}
    for base, _dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(base, fname)
            try:
                with open(path, encoding="utf-8") as fh:
                    layers = json.load(fh).get("layers") or []
            except (OSError, ValueError, AttributeError):
                continue
            parts = os.path.relpath(path, root).replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            tag = "%s:%s" % (parts[-2], parts[-1])
            if parts[0] != "registry.ollama.ai" or parts[1] != "library":
                tag = "/".join(parts[1:-1]) + ":" + parts[-1]
            for layer in layers:
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    blob = str(layer.get("digest", "")).replace(":", "-")
                    names.setdefault(blob, tag)
    return names


def classify(rows, session=None, blob_names=None):
    """Turn raw rows into Procs with a description and a leftover flag."""
    session = session if session is not None else state.load_session()
    servers = ((session.get("llamacpp") or {}).get("servers") or {})
    zoomies_ports = {int(r.get("port") or 0): r for r in servers.values()}
    zoomies_shells = {int(r.get("shell_pid") or 0) for r in servers.values()}
    script_dir = os.path.normcase(state.SCRIPT_DIR)

    procs = [Proc(pid=int(r["pid"]), name=str(r.get("name", "")),
                  cmd=str(r.get("cmd") or ""), ram=int(r.get("ram") or 0),
                  started=str(r.get("started") or ""), ppid=int(r.get("ppid") or 0),
                  parent=str(r.get("parent") or ""),
                  parent_cmd=str(r.get("parent_cmd") or ""),
                  ports=[int(p) for p in (r.get("ports") or []) if p])
             for r in rows]
    by_pid = {p.pid: p for p in procs}

    def is_router(p):
        return (p.name.lower() in ("llama.exe", "llama-server.exe")
                and _subcommand(p.cmd) in ("serve", "")
                and not _flag(p.cmd, MODEL_FLAGS))

    routers = {p.pid for p in procs if is_router(p)}
    ollama_servers = {p.pid for p in procs
                      if p.name.lower() == "ollama.exe" and _subcommand(p.cmd) == "serve"}

    for p in procs:
        low = p.name.lower()
        gone = " - whatever started it has exited" if not p.parent else ""
        from_shell = p.parent.lower() in SHELLS
        port_text = (" on port %s" % ",".join(map(str, p.ports))) if p.ports else ""

        if low == "ollama app.exe":
            p.what, p.protected = "Ollama tray app", True
        elif low == "ollama.exe":
            sub = _subcommand(p.cmd)
            if sub == "serve":
                p.what, p.protected = "Ollama server", True
            elif sub == "runner":
                blob = os.path.basename(_flag(p.cmd, ("--model",)))
                if blob_names is None:
                    blob_names = ollama_blob_names()
                model = blob_names.get(blob, blob[:19] or "a model")
                if p.ppid in ollama_servers:
                    p.what = "Ollama model: %s (Ollama unloads it)" % model
                else:
                    p.what = "Ollama model: %s - its Ollama server is gone" % model
                    p.leftover = True
            elif from_shell:
                args = re.sub(r'^\s*("[^"]*"|\S+)\s*', "", p.cmd, count=1)
                p.what = "Ollama command in a terminal: ollama %s" % args[:60]
                p.leftover = True
            else:
                p.what = "Ollama helper%s" % gone
        else:                                             # llama.exe / llama-server.exe
            model = _flag(p.cmd, ("--alias",) + MODEL_FLAGS)
            sub = _subcommand(p.cmd)
            ours_port = next((port for port in p.ports if port in zoomies_ports), 0)
            if p.ppid in routers:
                p.what = "llama.cpp app model: %s" % (model or "?")
            elif ours_port or (p.ppid in zoomies_shells and p.ppid):
                port = ours_port or next(iter(p.ports), 0)
                rec = zoomies_ports.get(port) or {}
                p.what = "Zoomies server: %s%s" % (rec.get("label") or model or "?", port_text)
                p.zoomies_port = port
            elif script_dir and script_dir in os.path.normcase(p.parent_cmd):
                p.what = "Zoomies server it lost track of: %s%s" % (model or "?", port_text)
                p.leftover = True
            elif p.pid in routers:
                p.what = "llama.cpp server with no model of its own%s%s" % (port_text, gone)
            elif sub == "cli":
                p.what = "llama.cpp chat in a terminal: %s%s" % (model or "?", gone)
                p.leftover = True
            else:
                p.what = "llama.cpp server started outside Zoomies: %s%s%s" % (
                    model or "?", port_text, gone)
                p.leftover = True
    order = {"ollama app.exe": 0, "ollama.exe": 1, "llama.exe": 2, "llama-server.exe": 2}
    procs.sort(key=lambda p: (p.leftover, order.get(p.name.lower(), 3), p.started))
    return procs


def listing(session=None):
    return classify(snapshot(), session)


def end(proc, session=None):
    """End one process tree. Returns (ok, message).

    Re-reads the process table first: the row on screen may be minutes old,
    and a pid Windows has since handed to another program must not be hit.
    """
    if proc.protected:
        return False, "%s is never ended from Zoomies" % proc.what
    if proc.name.lower() not in NAMES:
        return False, "not a llama.cpp or Ollama process"
    current = {int(r["pid"]): r for r in snapshot()}
    row = current.get(proc.pid)
    if not row:
        return True, "already gone"
    if str(row.get("name", "")).lower() != proc.name.lower() \
            or str(row.get("started") or "") != proc.started:
        return False, "pid %d now belongs to a different process - left alone" % proc.pid
    try:
        res = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                             capture_output=True, text=True, timeout=30,
                             creationflags=state.CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if res.returncode != 0:
        return False, (res.stderr or res.stdout or "taskkill failed").strip()
    if proc.zoomies_port and session is not None:
        ((session.get("llamacpp") or {}).get("servers") or {}).pop(
            str(proc.zoomies_port), None)
    return True, "ended"


def fmt_ram(n):
    if not n:
        return "-"
    return "%.1f GB" % (n / 1024.0 ** 3) if n >= 1024 ** 3 else "%d MB" % (n / 1024 ** 2)
