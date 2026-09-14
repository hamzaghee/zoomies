r"""
Zoomies - live inference metrics.

Ported from the Ollama Monitor, generalised to both backends.

Ollama and Unsloth both run llama.cpp underneath, and llama.cpp writes the
same `slot print_timing:` lines either way - Unsloth just prefixes each one
with a timestamp. Since the patterns are searched rather than anchored, one
parser serves both. What differs is only where the log lives:

  Ollama    %LOCALAPPDATA%\Ollama\server.log        one fixed file
  Unsloth   ~\.unsloth\studio\logs\llama-server\    a new file per run, named
                                                    with a random inner port

so the Unsloth source re-resolves to the newest file as it goes.

GPU utilisation comes from Windows' own "GPU Engine" counters - the same
source Task Manager reads - joined to real adapter names through DXGI.
"""

import csv
import io
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime

import state

CREATE_NO_WINDOW = 0x08000000


def _try_remove(path, attempts=6, pause=0.15):
    """Delete a file, tolerating Windows holding it open a moment longer.

    Killing typeperf does not release its handle instantly, so a single
    os.remove straight afterwards loses the race and fails. Swallowing that
    failure is how sample files pile up in %TEMP% forever.
    """
    for i in range(attempts):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if i + 1 < attempts:
                time.sleep(pause)
    return False

# --- llama.cpp slot timing lines ------------------------------------------
RE_NEW_PROMPT = re.compile(
    r"new prompt, n_ctx_slot = (?P<n_ctx>\d+), n_keep = \d+, "
    r"task\.n_tokens = (?P<n_tokens>\d+)")
RE_PROMPT_PROGRESS = re.compile(
    r"prompt processing, n_tokens\s*=\s*(?P<n_tokens>\d+), "
    r"progress = (?P<progress>[\d.]+), "
    r"t\s*=\s*(?P<t>[\d.]+) s / (?P<tps>[\d.]+) tokens per second")
RE_GEN = re.compile(
    r"n_gen\s*=\s*(?P<n_gen>\d+), tg\s*=\s*(?P<tg>[\d.]+) t/s, "
    r"tg_3s\s*=\s*(?P<tg3s>[\d.]+) t/s")

LUID_RE = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)")

GPU_POLL_SECONDS = 2
GPU_SAMPLE_PREFIX = "zoomies_gpu_"
HISTORY_MAXLEN = 25
IDLE_AFTER = 5.0            # seconds of quiet before "Idle" is shown


# --------------------------------------------------------------------------
# where the logs are
# --------------------------------------------------------------------------

def ollama_log():
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama",
                        "server.log")


def unsloth_log():
    """Newest llama-server log. Unsloth starts a fresh one per run, so the
    answer changes underneath us and has to be re-checked."""
    folder = os.path.join(os.path.expanduser("~"), ".unsloth", "studio",
                          "logs", "llama-server")
    try:
        entries = [(os.path.getmtime(os.path.join(folder, n)),
                    os.path.join(folder, n))
                   for n in os.listdir(folder) if n.endswith(".log")]
    except OSError:
        return ""
    return max(entries)[1] if entries else ""


SOURCES = (("ollama", ollama_log), ("unsloth", unsloth_log))


# --------------------------------------------------------------------------
# the collector
# --------------------------------------------------------------------------

class Metrics:
    """Live numbers for whichever backend is talking.

    Everything here runs on worker threads and is published through one
    lock-guarded dict. Nothing in this module touches a tk widget; the GUI
    calls snapshot() from its own refresh loop.
    """

    def __init__(self, shutdown, model_namer=None, sources=None):
        self.shutdown = shutdown
        # Injectable so the parser can be driven over known files
        # in a test instead of whatever the machine happens to hold.
        self.sources = sources or SOURCES
        self.model_namer = model_namer or (lambda backend: "")
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_MAXLEN)
        self.threads = []
        self._sampler_lock = threading.Lock()
        self._sampler = None
        self._our_samples = set()
        self.state = {
            "status": "Waiting for activity...",
            "backend": "",
            "n_ctx": None, "n_tokens": None,
            "prompt_tps": None, "prompt_progress": None,
            "n_gen": None, "tg": None, "tg3s": None, "ttft": None,
            "last_update": None, "request_start": None,
            "first_gen_seen": False, "filed": False,
            "gpus": [], "gpu_error": "",
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        for name, resolver in self.sources:
            t = threading.Thread(target=self._parse_loop, args=(name, resolver),
                                 daemon=True, name="metrics-" + name)
            t.start()
            self.threads.append(t)
        t = threading.Thread(target=self._gpu_loop, daemon=True, name="metrics-gpu")
        t.start()
        self.threads.append(t)
        return self

    def stop(self):
        self.shutdown.set()
        self._kill_sampler()
        for t in self.threads:
            t.join(timeout=2)
        self.cleanup()
        if self._our_samples:        # a handle that was still closing
            time.sleep(0.4)
            self.cleanup()

    def snapshot(self):
        with self.lock:
            snap = dict(self.state)
            snap["history"] = list(self.history)
        last = snap.get("last_update")
        if last and time.time() - last > IDLE_AFTER and \
                snap["status"] not in ("Waiting for activity...",):
            snap["status"] = "Idle"
        return snap

    # -- log parsing -------------------------------------------------------

    def _follow(self, resolver):
        """Yield new lines, re-resolving when the source file is replaced.

        Starts at the end of the file: replaying a log that is already
        megabytes long would flood the history with ancient requests.
        """
        handle, path, checked = None, "", 0.0
        try:
            while not self.shutdown.is_set():
                now = time.time()
                if now - checked > 2.0:
                    checked = now
                    latest = resolver()
                    if latest and latest != path:
                        if handle:
                            handle.close()
                            handle = None
                        path = latest
                if handle is None:
                    if not path or not os.path.isfile(path):
                        if self.shutdown.wait(1.0):
                            return
                        continue
                    try:
                        handle = open(path, "r", encoding="utf-8", errors="ignore")
                        handle.seek(0, os.SEEK_END)
                    except OSError:
                        if self.shutdown.wait(1.0):
                            return
                        continue
                line = handle.readline()
                if line:
                    yield line.rstrip("\n")
                elif self.shutdown.wait(0.1):
                    return
        finally:
            if handle:
                try:
                    handle.close()
                except OSError:
                    pass

    def _parse_loop(self, backend, resolver):
        for line in self._follow(resolver):
            now = time.time()

            m = RE_NEW_PROMPT.search(line)
            if m:
                with self.lock:
                    self._push_history(backend)
                    self.state.update({
                        "status": "Processing prompt...", "backend": backend,
                        "n_ctx": int(m.group("n_ctx")),
                        "n_tokens": int(m.group("n_tokens")),
                        "prompt_tps": None, "prompt_progress": 0.0,
                        "n_gen": None, "tg": None, "tg3s": None, "ttft": None,
                        "request_start": now, "first_gen_seen": False,
                        "last_update": now, "filed": False,
                    })
                continue

            m = RE_PROMPT_PROGRESS.search(line)
            if m:
                with self.lock:
                    self.state.update({
                        "status": "Processing prompt...", "backend": backend,
                        "n_tokens": int(m.group("n_tokens")),
                        "prompt_progress": float(m.group("progress")),
                        "prompt_tps": float(m.group("tps")),
                        "last_update": now,
                    })
                continue

            m = RE_GEN.search(line)
            if m:
                with self.lock:
                    # Time to first token is wall clock between the prompt
                    # arriving and the first generation line, because
                    # llama.cpp does not report it directly.
                    if not self.state["first_gen_seen"] and self.state["request_start"]:
                        self.state["ttft"] = now - self.state["request_start"]
                        self.state["first_gen_seen"] = True
                    self.state.update({
                        "status": "Generating...", "backend": backend,
                        "n_gen": int(m.group("n_gen")),
                        "tg": float(m.group("tg")),
                        "tg3s": float(m.group("tg3s")),
                        "last_update": now,
                    })
                continue

    def _push_history(self, backend):
        """Called with the lock held. Files the current request into history.

        Idempotent via the "filed" flag, because there are two callers: the
        arrival of the next prompt, and the idle reaper. The original only
        had the first, which meant the last request of a session never
        appeared in the table - you had to send another prompt to see the
        previous one.
        """
        s = self.state
        if s.get("n_gen") is None or s.get("filed"):
            return                      # nothing generated, or already filed
        s["filed"] = True
        self.history.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "backend": s.get("backend") or backend,
            "model": self.model_namer(s.get("backend") or backend),
            "ttft": s.get("ttft"), "tg3s": s.get("tg3s"),
            "n_gen": s.get("n_gen"), "n_tokens": s.get("n_tokens"),
            "n_ctx": s.get("n_ctx"),
        })

    # -- GPU ---------------------------------------------------------------

    def _kill_sampler(self):
        with self._sampler_lock:
            proc = self._sampler
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass

    def cleanup(self):
        """Remove any sample files we left behind, plus stale ones from a
        previous run that died before it could tidy up."""
        with self._sampler_lock:
            mine = list(self._our_samples)
        for path in mine:
            if _try_remove(path, attempts=8):
                with self._sampler_lock:
                    self._our_samples.discard(path)
        cutoff = time.time() - 300
        folder = tempfile.gettempdir()
        try:
            names = os.listdir(folder)
        except OSError:
            return
        for name in names:
            if not name.startswith(GPU_SAMPLE_PREFIX):
                continue
            path = os.path.join(folder, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass        # held by another instance, or already gone

    def sample_gpu(self):
        """{luid: max utilisation percent}.

        Windows reports one value per engine per process; Task Manager shows
        the highest per adapter rather than the sum, and so do we - summing
        would double count engines running in parallel.

        typeperf writes to a file instead of a pipe for two reasons learned
        the hard way: without a console it wraps long CSV rows and corrupts
        them, and an unread pipe is how an orphaned typeperf wedges forever.
        "Utilization Percentage" is a rate counter, so the first of the two
        samples is always blank and only the second is usable.
        """
        if self.shutdown.is_set():
            return {}, ""
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            "%s%s.csv" % (GPU_SAMPLE_PREFIX, uuid.uuid4().hex))
        with self._sampler_lock:
            self._our_samples.add(tmp_path)
        try:
            cmd = ["typeperf", r"\GPU Engine(*)\Utilization Percentage",
                   "-sc", "2", "-f", "CSV", "-o", tmp_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True,
                                    creationflags=CREATE_NO_WINDOW)
            with self._sampler_lock:
                self._sampler = proc
                if self.shutdown.is_set():
                    proc.kill()
            try:
                _out, errors = proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                return {}, "typeperf timed out"
            finally:
                with self._sampler_lock:
                    self._sampler = None
            if proc.returncode != 0:
                return {}, (errors or "").strip()[:120] or "typeperf failed"

            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as fh:
                rows = [r for r in csv.reader(io.StringIO(fh.read())) if r]
            if len(rows) < 3:                 # header plus two samples
                return {}, "no counter data"
            headers, values = rows[0], rows[-1]
            per_luid = {}
            for header, value in zip(headers, values):
                m = LUID_RE.search(header)
                if not m:
                    continue
                # upper-cased so it joins cleanly against DXGI, which always
                # formats LUIDs in upper-case hex
                luid = (m.group(1) + "_" + m.group(2)).upper()
                try:
                    pct = float(value)
                except ValueError:
                    continue
                per_luid[luid] = max(per_luid.get(luid, 0.0), pct)
            return per_luid, ""
        except OSError as exc:
            return {}, str(exc)[:120]
        finally:
            # Only stop tracking it once it is actually gone - otherwise a
            # file we failed to delete is forgotten and never retried.
            if _try_remove(tmp_path):
                with self._sampler_lock:
                    self._our_samples.discard(tmp_path)

    def _gpu_loop(self):
        adapters = []
        refreshed = 0.0
        while not self.shutdown.is_set():
            now = time.time()
            if not adapters or now - refreshed > 30:
                adapters = [a for a in state.enumerate_adapters()
                            if not a["is_software"] and a["vram"] >= 1 << 30]
                refreshed = now
            usage, err = self.sample_gpu()
            rows = []
            for adapter in adapters:
                rows.append({
                    "name": state.display_label(adapter, adapters),
                    "luid": adapter["luid"],
                    "vram": adapter["vram"],
                    "pct": usage.get(adapter["luid"]),
                })
            with self.lock:
                self.state["gpus"] = rows
                self.state["gpu_error"] = err
                # File a finished request once it has gone quiet, so the last
                # one of a session still shows up in the table.
                # time.time() again, not the `now` from the top of the loop:
                # sample_gpu() blocks for about three seconds, so that value
                # is stale by the time we get here and the request would look
                # more recent than it is.
                last = self.state.get("last_update")
                if (self.state.get("n_gen") is not None
                        and not self.state.get("filed")
                        and last and time.time() - last > IDLE_AFTER):
                    self._push_history(self.state.get("backend") or "")
            self.shutdown.wait(GPU_POLL_SECONDS)


# --------------------------------------------------------------------------
# formatting helpers for the GUI
# --------------------------------------------------------------------------

def fmt(value, suffix="", digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return ("%%.%df%%s" % digits) % (value, suffix)
    return "%s%s" % (value, suffix)


def fmt_int(value):
    return "-" if value is None else format(int(value), ",")
