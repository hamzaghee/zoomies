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
import ctypes
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
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

# Printed once a request finishes - exact averages, better than anything we
# can derive. "eval time" also appears inside "prompt eval time", so the
# generation pattern must refuse a match preceded by "prompt ".
RE_PROMPT_EVAL = re.compile(
    r"prompt eval time =\s*(?P<ms>[\d.]+) ms /\s*(?P<n>\d+) tokens"
    r".*?(?P<tps>[\d.]+) tokens per second")
RE_EVAL = re.compile(
    r"(?<!prompt )eval time =\s*(?P<ms>[\d.]+) ms /\s*(?P<n>\d+) tokens"
    r".*?(?P<tps>[\d.]+) tokens per second")
RE_TOTAL = re.compile(r"total time =\s*(?P<ms>[\d.]+) ms")
RE_KV = re.compile(r"llama_kv_cache:.*?K \((?P<k>[^)]+)\).*?V \((?P<v>[^)]+)\)")

LUID_RE = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)")

GPU_POLL_SECONDS = 2
SLOTS_POLL_SECONDS = 0.5      # how often each llama-server is asked
PORT_REFRESH_SECONDS = 5.0    # how often to look for new llama-server ports
SLOTS_FRESH = 3.0             # slots data this recent outranks log lines
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
# asking llama-server directly
# --------------------------------------------------------------------------
#
# Logs only work if whoever started the backend kept them. jobbuddy starts
# `ollama serve` with stderr sent to DEVNULL, so server.log is never written
# and a log-only monitor sits on "Waiting for activity" while a model is
# clearly busy. Both backends run a llama-server process underneath, and
# llama-server answers GET /slots with live per-slot counters no matter who
# launched it - so that is the primary source, and logs are the fallback.

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def process_image(pid):
    """Full exe path for a pid, or ''. ctypes, so no subprocess per call."""
    try:
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_ulong(len(buf))
            if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value
        finally:
            k32.CloseHandle(handle)
    except (AttributeError, OSError, ValueError):
        pass
    return ""


def backend_for_image(path):
    """Which backend owns a llama-server, judged by where its exe lives."""
    low = (path or "").lower().replace("\\", "/")
    if "/ollama/" in low:
        return "ollama"
    if "unsloth" in low:
        return "unsloth"
    return "llama"


def llama_server_ports():
    """{port: (backend, pid)} for every llama-server on this machine.

    llama-server picks a random port each time a model loads, so it has to
    be discovered. netstat maps listening ports to pids in about 60 ms.
    """
    pids = set(state.find_processes("llama-server.exe"))
    if not pids:
        return {}
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True, timeout=15,
                             creationflags=CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3] != "LISTENING":
            continue
        try:
            pid = int(parts[4])
        except ValueError:
            continue
        if pid not in pids:
            continue
        address, _, port = parts[1].rpartition(":")
        if address not in ("127.0.0.1", "0.0.0.0", "[::1]", "[::]"):
            continue
        try:
            found[int(port)] = (backend_for_image(process_image(pid)), pid)
        except ValueError:
            continue
    return found


def parse_cache_types(cmdline):
    """KV cache quantisation from a llama-server command line.

    llama-server does not report its cache type on any endpoint, but the
    launcher always spells it out on the command line - Ollama turns the
    OLLAMA_KV_CACHE_TYPE environment variable into --cache-type-k/-v flags.
    No flag means llama.cpp's default, f16.
    """
    def find(flags):
        for flag in flags:
            m = re.search(r"(?:^|\s)%s(?:\s+|=)(\S+)" % re.escape(flag),
                          cmdline or "")
            if m:
                return m.group(1).strip('"')
        return "f16"
    k = find(("--cache-type-k", "-ctk"))
    v = find(("--cache-type-v", "-ctv"))
    return k if k == v else "K %s / V %s" % (k, v)


_KV_BY_PID = {}


def cache_types_for_pid(pid):
    """Looked up once per llama-server process - about a quarter of a second
    through WMI - and cached, since the flags cannot change while it runs."""
    if pid in _KV_BY_PID:
        return _KV_BY_PID[pid]
    script = ("(Get-CimInstance Win32_Process -Filter 'ProcessId=%d')"
              ".CommandLine" % int(pid))
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30,
            creationflags=CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    kv = parse_cache_types(out) if out.strip() else None
    _KV_BY_PID[pid] = kv
    return kv


def fetch_slots(port, timeout=1.0):
    req = urllib.request.Request("http://127.0.0.1:%d/slots" % port,
                                 headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


# --------------------------------------------------------------------------
# the collector
# --------------------------------------------------------------------------

class Metrics:
    """Live numbers for whichever backend is talking.

    Everything here runs on worker threads and is published through one
    lock-guarded dict. Nothing in this module touches a tk widget; the GUI
    calls snapshot() from its own refresh loop.
    """

    def __init__(self, shutdown, model_namer=None, sources=None,
                 port_finder=None):
        self.shutdown = shutdown
        # Both injectable so the collector can be driven over known inputs
        # in a test instead of whatever the machine happens to be running.
        self.sources = sources or SOURCES
        self.port_finder = port_finder or llama_server_ports
        self.model_namer = model_namer or (lambda backend: "")
        self.lock = threading.Lock()
        self.history = deque(maxlen=HISTORY_MAXLEN)
        self.threads = []
        self._sampler_lock = threading.Lock()
        self._sampler = None
        self._our_samples = set()
        self._slots_seen = {}         # backend -> last time /slots showed work
        self._tracks = {}             # (port, slot id) -> request tracker
        self.state = {
            "status": "Waiting for activity...",
            "backend": "",
            "n_ctx": None, "n_tokens": None,
            "prompt_tps": None, "prompt_progress": None,
            "n_gen": None, "tg": None, "tg3s": None, "ttft": None,
            "last_update": None, "request_start": None,
            "first_gen_seen": False, "filed": False,
            "gen_avg": None, "runtime": None, "kv": None,
            "gpus": [], "gpu_error": "",
        }

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        for name, resolver in self.sources:
            t = threading.Thread(target=self._parse_loop, args=(name, resolver),
                                 daemon=True, name="metrics-" + name)
            t.start()
            self.threads.append(t)
        t = threading.Thread(target=self._slots_loop, daemon=True,
                             name="metrics-slots")
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
            # /slots is watching this backend right now; the log would only
            # describe the same request a second time.
            if now - self._slots_seen.get(backend, 0) < SLOTS_FRESH:
                continue

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
                        "gen_avg": None, "runtime": None, "gen0": None,
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

            m = RE_KV.search(line)
            if m:
                k, v = m.group("k").strip(), m.group("v").strip()
                with self.lock:
                    self.state["kv"] = k if k == v else "K %s / V %s" % (k, v)
                continue

            m = RE_PROMPT_EVAL.search(line)
            if m:
                with self.lock:
                    self.state["prompt_tps"] = float(m.group("tps"))
                    self.state["last_update"] = now
                continue

            m = RE_EVAL.search(line)
            if m:
                with self.lock:          # exact run average, first to last token
                    self.state["gen_avg"] = float(m.group("tps"))
                    self.state["last_update"] = now
                continue

            m = RE_TOTAL.search(line)
            if m:
                with self.lock:
                    self.state["runtime"] = float(m.group("ms")) / 1000.0
                    self.state["last_update"] = now
                continue

            m = RE_GEN.search(line)
            if m:
                n_gen = int(m.group("n_gen"))
                with self.lock:
                    # Time to first token is wall clock between the prompt
                    # arriving and the first generation line, because
                    # llama.cpp does not report it directly.
                    if not self.state["first_gen_seen"] and self.state["request_start"]:
                        self.state["ttft"] = now - self.state["request_start"]
                        self.state["first_gen_seen"] = True
                    # Estimates until llama.cpp prints its exact end-of-request
                    # figures, which then overwrite these.
                    gen0 = self.state.get("gen0")
                    if gen0 is None:
                        self.state["gen0"] = (now, n_gen)
                    elif now > gen0[0] and n_gen > gen0[1]:
                        self.state["gen_avg"] = (n_gen - gen0[1]) / (now - gen0[0])
                    if self.state.get("request_start"):
                        self.state["runtime"] = now - self.state["request_start"]
                    self.state.update({
                        "status": "Generating...", "backend": backend,
                        "n_gen": n_gen,
                        "tg": float(m.group("tg")),
                        "tg3s": float(m.group("tg3s")),
                        "last_update": now,
                    })
                continue

    def _slots_loop(self):
        ports, scanned = {}, 0.0
        while not self.shutdown.is_set():
            now = time.time()
            if now - scanned > PORT_REFRESH_SECONDS:
                ports, scanned = self.port_finder(), now
            for port, (backend, pid) in ports.items():
                kv = cache_types_for_pid(pid)
                for slot in fetch_slots(port):
                    self._observe_slot(port, backend, slot, time.time(), kv=kv)
            self._reap_idle()
            self.shutdown.wait(SLOTS_POLL_SECONDS)

    def _reap_idle(self):
        """File a finished request once it has gone quiet, so the last one of
        a session still shows up in the table.

        Runs on the half-second slots tick. It used to ride the GPU loop,
        whose typeperf sample blocks for three seconds or more under load,
        which made History appear late and at uneven intervals.
        """
        with self.lock:
            last = self.state.get("last_update")
            if (self.state.get("n_gen") is not None
                    and not self.state.get("filed")
                    and last and time.time() - last > IDLE_AFTER):
                self._push_history(self.state.get("backend") or "")

    def _observe_slot(self, port, backend, slot, now, kv=None):
        """Turn one /slots sample into the same state the log parser builds.

        n_prompt_tokens already includes generated tokens (prompt + decoded),
        so it is the context in use. Speeds come from differences between
        samples, so TTFT is accurate to about one poll interval.
        """
        key = (port, slot.get("id", 0))
        tokens = slot.get("next_token") or [{}]
        first = tokens[0] if isinstance(tokens, list) and tokens else tokens
        decoded = int((first or {}).get("n_decoded") or 0)
        task = slot.get("id_task")
        busy = bool(slot.get("is_processing"))
        track = self._tracks.get(key)

        with self.lock:
            if not busy:
                if track is not None:
                    self._push_history(backend)
                    self._tracks.pop(key, None)
                return

            if track is None or track["task"] != task:
                if track is not None:
                    self._push_history(backend)
                track = {"task": task, "start": now, "first_tok": None,
                         "samples": deque(maxlen=40), "prompt_done": 0,
                         "prompt_t": now,
                         # prompt-phase baseline, only if we arrived before
                         # generation started; otherwise the average is unknown
                         "p0": None, "pt0": None,
                         # generation: first token, previous sample, latest token
                         "t_first": None, "d_first": None,
                         "t_prev": None, "d_prev": None,
                         "t_last": None, "d_last": None}
                self._tracks[key] = track
                self.state.update({"n_gen": None, "tg": None, "tg3s": None,
                                   "ttft": None, "prompt_tps": None,
                                   "request_start": now, "filed": False,
                                   "first_gen_seen": False,
                                   "gen_avg": None, "runtime": None})

            processed = int(slot.get("n_prompt_tokens_processed") or 0)
            n_prompt = int(slot.get("n_prompt_tokens") or 0)
            prompt_total = max(1, n_prompt - decoded)
            if decoded == 0:
                # Average over the whole watched prompt phase, not the jump
                # between two samples: llama.cpp processes prompts in batches,
                # so sample-to-sample rates swing between zero and huge.
                if track["p0"] is None:
                    track["p0"], track["pt0"] = processed, now
                elif processed > track["p0"] and now > track["pt0"]:
                    self.state["prompt_tps"] = ((processed - track["p0"])
                                                / (now - track["pt0"]))
                track["prompt_done"], track["prompt_t"] = processed, now
            elif (track["first_tok"] is None and track["p0"] is not None
                    and self.state.get("prompt_tps") is None
                    and prompt_total > track["p0"] and now > track["pt0"]):
                # The prompt finished between two samples, so there was never
                # a second prompt-phase reading. It was certainly done by now,
                # which makes this a slight underestimate rather than a guess.
                self.state["prompt_tps"] = ((prompt_total - track["p0"])
                                            / (now - track["pt0"]))

            update = {"backend": backend, "last_update": now,
                      "n_ctx": slot.get("n_ctx"), "n_tokens": n_prompt,
                      "prompt_progress": min(1.0, processed / prompt_total)}
            if kv:
                update["kv"] = kv
            if decoded > 0:
                if track["first_tok"] is None:
                    track["first_tok"] = now
                    # Only meaningful if the prompt phase was actually
                    # watched. A request first seen mid-generation has an
                    # unknown start, and "0.00s" would be a made-up number.
                    if track["prompt_done"]:
                        update["ttft"] = now - track["start"]
                    update["first_gen_seen"] = True
                    track["t_first"], track["d_first"] = now, decoded
                # Current rate: since the previous sample, stalls included.
                if track["t_prev"] is not None and now > track["t_prev"]:
                    update["tg"] = (decoded - track["d_prev"]) / (now - track["t_prev"])
                track["t_prev"], track["d_prev"] = now, decoded
                # Run average: first token to the latest token. The end is the
                # last time the count actually rose, so time spent idle after
                # the final token does not dilute it.
                if track["d_last"] is None or decoded > track["d_last"]:
                    track["t_last"], track["d_last"] = now, decoded
                if track["t_last"] > track["t_first"]:
                    update["gen_avg"] = ((track["d_last"] - track["d_first"])
                                         / (track["t_last"] - track["t_first"]))
                # Runtime only if the request was watched from its start.
                if track["p0"] is not None:
                    update["runtime"] = track["t_last"] - track["start"]
                update["status"] = "Generating..."
                update["n_gen"] = decoded
            else:
                update["status"] = "Processing prompt..."
            self.state.update(update)
            self._slots_seen[backend] = now

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
            "prompt_tps": s.get("prompt_tps"),
            "gen_avg": s.get("gen_avg"), "runtime": s.get("runtime"),
            "kv": s.get("kv"),
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
            # A driver reinstall or TDR leaves a stale entry for a card that
            # is still installed: same hardware key, a new LUID, and no
            # counters behind it. Keep the one that is actually reporting.
            adapters = state.physical_adapters(adapters, live_luids=usage)
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


BAR_FULL, BAR_EMPTY = "\u2588", "\u2591"


def bar_text(used, total, width=12):
    """A text bar for a table cell, where no real widget can go."""
    if not used or not total:
        return "-"
    filled = max(0, min(width, int(width * used / float(total) + 0.5)))
    return "%s%s  %s / %s" % (BAR_FULL * filled, BAR_EMPTY * (width - filled),
                              format(int(used), ","), format(int(total), ","))


def fmt_mmss(seconds):
    if seconds is None:
        return "-"
    whole = int(round(seconds))
    return "%02d:%02d" % (whole // 60, whole % 60)


def fmt_int(value):
    return "-" if value is None else format(int(value), ",")
