r"""Zoomies - asking llama.cpp how big its compute buffer will be.

Every other part of the VRAM estimate comes out of the .gguf header, and all
of them were checked against llama.cpp's own figures to the byte: the
offloaded weights are tensor sizes, the KV cache is head counts times
context, the recurrent state is the convolution window plus the state
matrix. The compute buffer is the one part the header cannot answer, because
it is the peak of a graph llama.cpp builds differently for every
architecture. A formula fitted exactly to the Qwen 3.5 family lands 12% out
on Devstral and 20% out on Gemma 4, and there is no reason to expect the
next architecture to be kinder.

So it is not guessed. llama.cpp works it out itself, before it loads
anything: `-fit` projects what a load will need and prints it as a table.
`--device none` keeps that projection off the graphics cards entirely, and
the whole thing takes about half a second - the model file is never read
past its header.

    | memory breakdown [MiB] | total free  self   model  context  compute ... |
    |   - Host               |            13104 = 10082 +   2622 +     400    |

What the table gives, measured over Qwen3.6-35B-A3B, Qwen3.8-27B, Ornith
1.5, Devstral 2 and Gemma 4, at two batch sizes and two contexts each:

    compute = ubatch x (graph per token + mask per token)

The compute buffer is exactly proportional to the micro-batch, and grows
with the *per-slot* context by the attention mask alone - 2 bytes a token
with flash attention on, and n_head x 4 with it off, which is why turning it
off at 131k context costs 8.5 GB instead of 0.25. "Graph per token" is what
is left once the mask is taken out, and it moves with neither the context
nor the batch, so one probe per model file answers every setting. It is
cached by path, size and modified time, like the header itself.

This module only ever runs llama.cpp with `--device none`: a probe must not
be able to take memory from the cards it is reporting on, or from a server
already running on them.
"""

import os
import re
import socket
import subprocess
import threading

import state

CACHE_PATH = os.path.join(state.CACHE_DIR, "fit.json")
CACHE_VERSION = 1

MiB = 1024 ** 2

# What to probe at. Small enough that the mask term is a rounding error next
# to the graph term, and that a fit on a 20 GB file still finishes in under a
# second; large enough to clear llama.cpp's own -fitc floor of 4096, below
# which it is allowed to move the context and report something else.
PROBE_CTX = 4096
PROBE_UB = 512

# The fit prints before the weights are touched. This is how long to wait for
# it, not how long a load takes.
PROBE_TIMEOUT = 90.0

# "|   - Host   |   13104 = 10082 +   2622 +   400   |", and the same shape
# for a device row. Only the three summands are wanted.
_ROW = re.compile(r"^\s*\|\s*-\s*(\S+)\s*\|(.*?)\|\s*$")
_SUMS = re.compile(r"(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)")
_DONE = re.compile(r"fitting params to free memory took|"
                   r"failed to fit params|loading model tensors")

_lock = threading.Lock()
_cache = None


def _key(path):
    st = os.stat(path)
    return "%s|%d|%d" % (os.path.normcase(path), st.st_size, int(st.st_mtime))


def _load():
    global _cache
    if _cache is None:
        saved = state.read_json(CACHE_PATH, {}) or {}
        _cache = (saved.get("files") or {}) \
            if saved.get("version") == CACHE_VERSION else {}
    return _cache


def _probe_argv(exe, prefix):
    """(argv prefix, extra args) for a probe that loads nothing and listens
    on nothing.

    llama.exe is a multi-tool, and its `cli` subcommand fits its parameters
    the same way `serve` does without binding a port. llama-server.exe has
    only the server, which binds before it fits, so it is given a port the
    OS has just confirmed is free rather than the default one - which is
    very likely the port a model is already being served on.
    """
    if list(prefix) == ["serve"]:
        return ["cli"], ["-n", "0"]
    port = 0
    try:
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        finally:
            sock.close()
    except OSError:
        port = 18080
    return list(prefix), ["--port", str(port)]


def breakdown(exe, prefix, path, ctx=PROBE_CTX, ub=PROBE_UB, np=1,
              fa=True, extra=()):
    """llama.cpp's own projection for one load: {device: {model, context,
    compute}} in bytes, or {} when it could not be asked.

    Never raises. An estimate that has to wait on a subprocess still has to
    appear when the subprocess is not there.
    """
    if not exe or not path:
        return {}
    head, tail = _probe_argv(exe, prefix)
    argv = [exe] + head + [
        "-m", path, "-c", str(int(ctx)), "-np", str(int(np)),
        "-ub", str(int(ub)), "-fa", "on" if fa else "off",
        # The three that make this safe and quick: no graphics card is
        # touched, nothing is generated, and the memory table is printed at
        # all (it is trace-level output, which the default verbosity drops).
        "--device", "none", "--no-warmup", "-lv", "4",
    ] + list(extra) + tail
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True,
                                errors="replace", bufsize=1,
                                creationflags=state.CREATE_NO_WINDOW)
    except (OSError, ValueError):
        return {}
    rows, timer = {}, threading.Timer(PROBE_TIMEOUT, proc.kill)
    timer.start()
    try:
        for line in proc.stderr:
            row = _ROW.match(line[line.find("|"):] if "|" in line else "")
            if row:
                sums = _SUMS.search(row.group(2))
                if sums:
                    rows[row.group(1)] = {
                        "total": int(sums.group(1)) * MiB,
                        "model": int(sums.group(2)) * MiB,
                        "context": int(sums.group(3)) * MiB,
                        "compute": int(sums.group(4)) * MiB,
                    }
            if _DONE.search(line):
                break
    except (OSError, ValueError):
        pass
    finally:
        timer.cancel()
        proc.kill()
        try:
            proc.stderr.close()
            proc.wait(timeout=10)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return rows


def graph_per_token(path, exe, prefix=(), refresh=False):
    """Bytes of compute buffer per micro-batch token, apart from the
    attention mask - the one number the header cannot give.

    None when llama.cpp could not be asked, which the caller has to have a
    fallback for; the answer is cached on disk either way, so a model that
    has been probed once never pays for it again.
    """
    try:
        key = _key(path)
    except OSError:
        return None
    with _lock:
        cache = _load()
        if not refresh and key in cache:
            return cache[key].get("graph_per_token")
    rows = breakdown(exe, prefix, path)
    host = rows.get("Host") or rows.get("CPU")
    if not host:
        return None
    # Everything but the mask, which at the probe's own context is small and
    # exactly known: 2 bytes a token with flash attention on.
    per_token = host["compute"] / float(PROBE_UB) - 2 * PROBE_CTX
    if per_token <= 0:
        return None
    entry = {"graph_per_token": int(per_token),
             "probe_compute": host["compute"],
             "probe_ctx": PROBE_CTX, "probe_ub": PROBE_UB}
    with _lock:
        cache = _load()
        cache[key] = entry
        state.write_json(CACHE_PATH, {"version": CACHE_VERSION,
                                      "files": cache})
    return entry["graph_per_token"]
