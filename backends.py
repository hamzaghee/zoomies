r"""
Zoomies - the backend seam.

This is the one module allowed to know what "Ollama" or "llama.cpp" means.
Everything else in the app talks to REGISTRY and the six methods on Backend,
so adding a third backend later is a change to this file only.

A note on the Ollama design, because it is not obvious:

  Ollama has no way to attach sampling settings to a model at launch.
  `ollama run` takes no such flags, /set parameter lives and dies inside the
  interactive REPL, and /api/generate options apply to one request only.
  See https://github.com/ollama/ollama/issues/5362 - open since June 2024.

  So to make "apply optimal settings" actually true, Zoomies creates a derived
  tag (base name + "-zoomies") carrying the settings, loads that, and deletes
  it again on unload. Ollama stores models as content-addressed blobs, so the
  derived tag reuses the base model's weights layer by digest - Ollama's own
  API replies "using already created layer sha256:..." while doing it. The
  cost is a manifest of roughly a kilobyte, not a copy of the weights.

  Everything here goes over HTTP. The only CLI call Zoomies ever makes is
  `ollama serve`, because every other ollama subcommand launches the Ollama
  tray application, which would break the promise that the backend's own GUI
  stays shut.
"""

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import runner
import state

# --------------------------------------------------------------------------
# shared shapes
# --------------------------------------------------------------------------

SETTING_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
    "presence_penalty", "seed", "context_length", "gpu_layers",
    "parallel", "flash_attn", "keep_alive", "extra_flags",
)

# Human labels for the settings panel.
SETTING_LABELS = (
    ("temperature", "Temperature"),
    ("top_p", "Top P"),
    ("top_k", "Top K"),
    ("min_p", "Min P"),
    ("repeat_penalty", "Repeat penalty"),
    ("presence_penalty", "Presence pen."),
    ("context_length", "Context"),
    ("seed", "Seed"),
    ("keep_alive", "Keep alive"),
    ("gpu_layers", "GPU layers"),
    ("parallel", "Parallel"),
    ("extra_flags", "Extra flags"),
)
SETTING_TEXT = dict(SETTING_LABELS)

# Explicit grid, rather than flowing the list into columns: sampling knobs on
# the first row, sizing on the second, backend-specific ones last, so the
# fields Ollama greys out sit together instead of being scattered.
SETTING_ROWS = (
    ("temperature", "top_p", "top_k", "min_p"),
    ("repeat_penalty", "presence_penalty", "context_length", "seed"),
    ("keep_alive", "gpu_layers", "parallel", None),
)
SETTING_WIDE = ("extra_flags",)


def slug(text):
    """Normalise a model name down to something we can match against a docs
    page title.

        qwen3.8:27b-q4_K_M              -> qwen3.8
        Qwen3.8 - How to Run Locally    -> qwen3.8
        gemma4:12b-it-q4_K_M            -> gemma4
        Gemma 4 - How to Run Locally    -> gemma4
        ministral-3:14b-instruct-2512   -> ministral3
        Ministral 3 - How to Run Guide  -> ministral3

    Squashing spaces is what makes "Gemma 4" and "gemma4" meet. Dots survive
    because the difference between qwen3 and qwen3.8 is a different model with
    different recommended settings, and conflating them would be silently wrong.
    """
    s = str(text).split(":")[0]
    s = re.split(r"\s+[-–—]\s+|\s*:\s+", s)[0]
    return re.sub(r"[^a-z0-9.]", "", s.lower())


@dataclass(frozen=True)
class ModelRecord:
    backend: str
    id: str                     # backend-native handle
    label: str                  # what the dropdown shows
    family: str = ""
    size_hint: str = ""         # "11.9B"
    quant: str = ""             # "Q4_K_M"
    context_max: int = 0        # 0 = unknown
    capabilities: tuple = ()
    size_bytes: int = 0
    source: str = "registry"    # "registry" | "folder"
    gguf_path: str = ""

    @property
    def match_key(self):
        return slug(self.id)

    def describe(self):
        bits = []
        if self.size_bytes:
            bits.append("%.1f GB" % (self.size_bytes / 1024.0 ** 3))
        if self.quant:
            bits.append(self.quant)
        if self.context_max:
            bits.append("%s ctx" % format(self.context_max, ","))
        caps = [c for c in self.capabilities if c != "completion"]
        if caps:
            bits.append(", ".join(caps))
        return "%s  -  %s" % (self.label, " · ".join(bits)) if bits else self.label


@dataclass
class LoadedModel:
    backend: str
    id: str
    label: str
    vram_bytes: int = 0
    context: int = 0
    endpoint: str = ""
    pid: int = 0
    expires: str = ""
    owned_by_us: bool = True
    log_path: str = ""
    is_derived: bool = False    # an Ollama "-zoomies" tag we created
    # Stopping it means ending a process Zoomies did not start, rather than
    # asking a manager that stays up to let the model go. Only ever on the
    # user's say-so: see _unload_others.
    foreign_process: bool = False


# --------------------------------------------------------------------------
# tiny HTTP client
# --------------------------------------------------------------------------

def http_json(url, method="GET", payload=None, timeout=5.0):
    """Return (data, error). Never raises - a dead backend is an expected
    state, not an exception, and the polling threads must not die on it."""
    data = None
    headers = {"Accept": "application/json", "User-Agent": "Zoomies/1.0"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace").strip()
        data = json.loads(raw) if raw else {}
        return data, ""
    except urllib.error.HTTPError as exc:
        return None, "HTTP %s" % exc.code
    except (urllib.error.URLError, OSError) as exc:
        return None, str(getattr(exc, "reason", exc))
    except ValueError:
        return None, "bad JSON from %s" % url


# --------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------

# llama.cpp's KV cache types, with its default spelled out first. One value
# sets both K and V.
KV_CACHE_DEFAULT = "f16 (default)"
KV_CACHE_CHOICES = (KV_CACHE_DEFAULT, "bf16", "q8_0", "q4_0", "q4_1",
                    "q5_0", "q5_1", "iq4_nl", "f32")
# What the dropdown starts on: q8_0 halves the cache of f16 at a quality cost
# too small to notice, and matches the q8_0 Ollama runs with here.
KV_CACHE_START = "q8_0"


def kv_choice_value(choice):
    """The cache type to send, or None for llama.cpp's own default."""
    value = str(choice or "").strip()
    if not value or value == KV_CACHE_DEFAULT or value not in KV_CACHE_CHOICES:
        return None
    return value


# llama.cpp refuses to start with one of these as the V cache unless flash
# attention is on; as the K cache they work either way.
KV_QUANTIZED = {"q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}


def flash_attn_off(tokens):
    """True if these llama-server flags turn flash attention off.

    The last -fa/--flash-attn wins, as it does in llama.cpp itself.
    """
    off = False
    for i, token in enumerate(tokens):
        name, eq, value = token.partition("=")
        if name not in ("-fa", "--flash-attn"):
            continue
        if not eq:
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
        off = value.strip().lower() == "off"
    return off


def ollama_kv_cache_type():
    """Ollama's KV cache type is fixed server-wide by OLLAMA_KV_CACHE_TYPE.

    Read from the registry as well as this process's environment, so a value
    set after Zoomies started is still seen. A user value overrides the
    machine one, the same precedence Windows itself applies.
    """
    value = os.environ.get("OLLAMA_KV_CACHE_TYPE", "")
    if not value:
        try:
            import winreg
            for root, path in (
                    (winreg.HKEY_CURRENT_USER, "Environment"),
                    (winreg.HKEY_LOCAL_MACHINE,
                     r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
                try:
                    with winreg.OpenKey(root, path) as key:
                        value = winreg.QueryValueEx(key, "OLLAMA_KV_CACHE_TYPE")[0]
                except OSError:
                    continue
                if value:
                    break
        except ImportError:
            pass
    return (value or "f16").strip()


def reasoning_kwargs(settings):
    """Chat-template kwargs for the chosen reasoning setting, or None.

    The style comes from the model's docs page, so a Gemma level can never be
    sent to a Qwen model: enable_thinking takes on/off, reasoning_effort takes
    whatever levels that page listed.
    """
    style = str(settings.get("reasoning_style") or "")
    level = str(settings.get("reasoning") or "").strip().lower()
    if not style or not level:
        return None
    if style == "enable_thinking":
        if level not in ("on", "off"):
            return None
        return {"enable_thinking": level == "on"}
    if style == "reasoning_effort":
        if level in ("none", "off"):
            # Thinking off is enable_thinking=false, never a "none" level.
            # Qwen3.8's docs list "none", but its chat template only accepts
            # xhigh/high/medium/low and raises on anything else - measured
            # live: reasoning_effort "none" turned every request into an
            # HTTP 500, while enable_thinking=false gave zero reasoning.
            return {"enable_thinking": False}
        return {"reasoning_effort": level}
    return None


class Backend:
    name = ""
    display_name = ""
    uses_model_folder = False
    default_folder = ""
    endpoint = ""
    host = "127.0.0.1"
    port = 0

    def __init__(self):
        self.last_error = ""

    def is_available(self):
        raise NotImplementedError

    def list_models(self, folder=None):
        raise NotImplementedError

    def supports(self, key):
        """Empty string means the backend cannot honour this setting, and the
        GUI greys the field out. Any other string is the reason, shown to the
        user, so nothing is ever silently ignored."""
        return ""

    def build_launch(self, model, settings, session, source_note=""):
        raise NotImplementedError

    def build_stop(self, loaded, session):
        raise NotImplementedError

    def list_loaded(self):
        return []

    def model_on_port(self, port):
        """The model a server on this port is holding, asked of the server
        itself, or "" if this backend cannot say."""
        return ""

    def fixed_kv_cache(self):
        """A KV cache type this backend imposes regardless of Zoomies, or None."""
        return None

    def remember_server(self, session, plan, shell_pid, model=None):
        """Record a server Zoomies started, so it can be found and stopped
        after a restart. What it is holding comes from the plan; `model` is
        only there for callers that still pass it."""

    def forget_server(self, session, plan):
        """Drop that record once the server is stopped."""


REGISTRY = {}


def register(backend):
    REGISTRY[backend.name] = backend
    return backend


def get(name):
    return REGISTRY.get(name)


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE = "http://%s:%d" % (OLLAMA_HOST, OLLAMA_PORT)

# Only these may go into an /api/create "parameters" block. Anything else is
# not a Modelfile parameter and Ollama would either ignore it or refuse.
OLLAMA_PARAM_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "seed": "seed",
    "context_length": "num_ctx",
}
INT_PARAMS = {"top_k", "seed", "num_ctx"}


def find_ollama_exe():
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Programs", "Ollama", "ollama.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama.exe"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    for directory in (os.environ.get("PATH") or "").split(os.pathsep):
        candidate = os.path.join(directory.strip('"'), "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
    return ""


def ollama_models_dir():
    return os.environ.get("OLLAMA_MODELS") or os.path.join(
        os.path.expanduser("~"), ".ollama", "models")


OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"


def ollama_manifests():
    """[(name, blob_path, size_bytes)] for every model Ollama has pulled.

    Read off disk rather than from /api/tags, because the point is the file:
    the weights are one content-addressed blob, and no Ollama endpoint says
    where it is. The layer marked .image.model is the GGUF itself; the
    projector, template and licence layers beside it are not loadable alone.

    Deliberately silent about anything unreadable - this feeds a dropdown, so
    one odd manifest must not cost the user the rest of the list.
    """
    root = os.path.join(ollama_models_dir(), "manifests")
    blob_dir = os.path.join(ollama_models_dir(), "blobs")
    out = []
    for base, _dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(base, fname)
            parts = os.path.relpath(path, root).replace("\\", "/").split("/")
            # <registry>/<namespace>/<model>/<tag>. Ollama leaves its own
            # registry and the "library" namespace out of the name it shows.
            parts = parts[1:]
            if parts and parts[0] == "library":
                parts = parts[1:]
            if len(parts) < 2:
                continue
            name = "%s:%s" % ("/".join(parts[:-1]), parts[-1])
            try:
                with open(path, encoding="utf-8") as fh:
                    layers = (json.load(fh) or {}).get("layers") or []
            except (OSError, ValueError, AttributeError):
                continue
            for layer in layers:
                if layer.get("mediaType") != OLLAMA_MODEL_LAYER:
                    continue
                blob = os.path.join(
                    blob_dir, str(layer.get("digest", "")).replace(":", "-"))
                if os.path.isfile(blob):
                    out.append((name, blob, int(layer.get("size") or 0)))
                break
    out.sort()
    return out


def is_gguf(path):
    """True if this file starts with llama.cpp's magic.

    Ollama's newer engine can store weights llama.cpp cannot read, under the
    same layer type in the same blob folder. Four bytes settle it, and
    offering a model that cannot load is worse than leaving it out.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def derived_tag(base):
    """Reserved namespace. Nothing without this suffix is ever deletable."""
    return base + state.TAG_SUFFIX


# ---- the safety rails ----------------------------------------------------

def tag_is_ours(tag, session):
    """Conditions 1, 2, 3 and 5 from the plan.

    Condition 4 (not currently loaded) is checked separately, because the stop
    script unloads before it deletes and so satisfies it by construction.

    Returns (ok, reason). A refusal is always explained, never silent.
    """
    if not tag:
        return False, "no tag given"
    if not tag.endswith(state.TAG_SUFFIX):
        return False, "%r does not end in %s" % (tag, state.TAG_SUFFIX)

    records = session.get("ollama", {}).get("created_tags", [])
    entry = next((r for r in records if r.get("tag") == tag), None)
    if entry is None:
        return False, "%r was not created by Zoomies" % tag
    if entry.get("pre_existing"):
        return False, "%r already existed before Zoomies touched it" % tag
    if entry.get("base") == tag:
        return False, "%r is a base model" % tag
    return True, ""


def can_remove_tag(tag, session, loaded_names):
    """All five conditions. Used by the startup sweep, where a leftover tag
    might still be loaded from a previous session."""
    ok, reason = tag_is_ours(tag, session)
    if not ok:
        return False, reason
    if tag in set(loaded_names or ()):
        return False, "%r is still loaded" % tag
    return True, ""


def record_created_tag(session, tag, base, pre_existing):
    records = session.setdefault("ollama", {}).setdefault("created_tags", [])
    records = [r for r in records if r.get("tag") != tag]
    records.append({
        "tag": tag,
        "base": base,
        "pre_existing": bool(pre_existing),
        "created": __import__("time").time(),
    })
    session["ollama"]["created_tags"] = records
    return session


def forget_tag(session, tag):
    records = session.get("ollama", {}).get("created_tags", [])
    session["ollama"]["created_tags"] = [r for r in records if r.get("tag") != tag]
    return session


class OllamaBackend(Backend):
    name = "ollama"
    display_name = "Ollama"
    uses_model_folder = False
    endpoint = OLLAMA_BASE
    host = OLLAMA_HOST
    port = OLLAMA_PORT

    def __init__(self):
        Backend.__init__(self)
        self.exe = find_ollama_exe()
        self.default_folder = ollama_models_dir()

    # -- availability ------------------------------------------------------

    def is_available(self):
        if not self.exe:
            return False, "ollama.exe not found"
        if state.port_open(OLLAMA_HOST, OLLAMA_PORT):
            return True, "running"
        return True, "installed, not running"

    def server_up(self):
        return state.port_open(OLLAMA_HOST, OLLAMA_PORT)

    # -- discovery ---------------------------------------------------------

    def list_models(self, folder=None):
        data, err = http_json(OLLAMA_BASE + "/api/tags", timeout=6.0)
        if err:
            self.last_error = err
            return self._list_from_manifests()
        self.last_error = ""
        out = []
        for entry in (data or {}).get("models", []) or []:
            name = entry.get("name") or entry.get("model") or ""
            if not name:
                continue
            details = entry.get("details") or {}
            caps = entry.get("capabilities") or details.get("capabilities") or []
            ctx = entry.get("context_length") or details.get("context_length") or 0
            out.append(ModelRecord(
                backend=self.name,
                id=name,
                label=name,
                family=details.get("family", ""),
                size_hint=details.get("parameter_size", ""),
                quant=details.get("quantization_level", ""),
                context_max=int(ctx or 0),
                capabilities=tuple(caps),
                size_bytes=int(entry.get("size") or 0),
                source="registry",
            ))
        out.sort(key=lambda m: m.label.lower())
        return out

    def _list_from_manifests(self):
        """Fallback for when the server is not running.

        Reads the manifests off disk. Deliberately does NOT shell out to
        `ollama list` to recover metadata: that would launch the tray app.
        """
        out = [ModelRecord(backend=self.name, id=name, label=name,
                           size_bytes=size, source="registry", gguf_path=blob)
               for name, blob, size in ollama_manifests()]
        out.sort(key=lambda m: m.label.lower())
        return out

    # -- capability matrix -------------------------------------------------

    def supports(self, key):
        if key in ("temperature", "top_p", "top_k", "min_p",
                   "repeat_penalty", "seed", "context_length"):
            return "applied via a temporary tag"
        if key == "presence_penalty":
            return "this load only"
        if key == "keep_alive":
            return "how long it stays in VRAM"
        # gpu_layers, parallel, flash_attn, extra_flags - and reasoning:
        # Ollama takes "think" per request, so whichever app sends the
        # prompt decides, and a launcher cannot set it for them.
        return ""

    # -- what is loaded ----------------------------------------------------

    def list_loaded(self):
        data, err = http_json(OLLAMA_BASE + "/api/ps", timeout=4.0)
        if err:
            self.last_error = err
            return []
        self.last_error = ""
        out = []
        for entry in (data or {}).get("models", []) or []:
            name = entry.get("name") or entry.get("model") or ""
            if not name:
                continue
            out.append(LoadedModel(
                backend=self.name,
                id=name,
                label=name,
                vram_bytes=int(entry.get("size_vram") or 0),
                context=int(entry.get("context_length") or 0),
                endpoint=OLLAMA_BASE,
                expires=str(entry.get("expires_at") or ""),
                owned_by_us=name.endswith(state.TAG_SUFFIX),
                is_derived=name.endswith(state.TAG_SUFFIX),
            ))
        out.sort(key=lambda m: m.label.lower())
        return out

    def fixed_kv_cache(self):
        return ollama_kv_cache_type()

    def installed_names(self):
        data, err = http_json(OLLAMA_BASE + "/api/tags", timeout=6.0)
        if err:
            return None                      # unknown, not "empty"
        return [m.get("name") or m.get("model") for m in (data or {}).get("models", [])]

    # -- script generation -------------------------------------------------

    def _params_block(self, settings):
        """Translate our normalised settings into Modelfile parameter names,
        skipping anything empty. An empty field means 'do not pass this', not
        'pass zero' - those are very different instructions to a sampler."""
        params = {}
        for key, target in OLLAMA_PARAM_MAP.items():
            raw = settings.get(key, "")
            if raw is None or str(raw).strip() == "":
                continue
            try:
                value = float(str(raw).replace(",", ""))
            except ValueError:
                continue
            params[target] = int(value) if target in INT_PARAMS else value
        return params

    def _preamble(self, log_path, title, model, source_note):
        return "\n".join([
            runner.header(title, model, source_note),
            "",
            "$ErrorActionPreference = 'Stop'",
            "$log  = %s" % runner.ps_single(log_path),
            "$base = %s" % runner.ps_single(OLLAMA_BASE),
            "$exe  = %s" % runner.ps_single(self.exe),
            "",
            "function Write-Log {",
            "  param($m)",
            "  $line = '[zoomies] ' + (Get-Date).ToString('o') + ' ' + $m",
            "  Write-Output $line",
            "  $line | Out-File -FilePath $log -Append -Encoding utf8",
            "}",
            "function Test-Srv {",
            "  try { Invoke-RestMethod \"$base/api/version\" -TimeoutSec 3 | Out-Null; return $true }",
            "  catch { return $false }",
            "}",
            "",
        ])

    def _ensure_server(self):
        return "\n".join([
            "if (-not (Test-Srv)) {",
            "  # 'serve' is the ONLY ollama subcommand Zoomies runs. Every other",
            "  # subcommand launches the Ollama tray app, which we promised not to open.",
            "  Write-Log 'Ollama is not responding - starting the server'",
            "  Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden",
            "  $n = 0",
            "  while (-not (Test-Srv) -and $n -lt 60) { Start-Sleep -Milliseconds 500; $n++ }",
            "  if (-not (Test-Srv)) { Write-Log 'server did not come up'; exit 3 }",
            "}",
            "Write-Log 'Ollama server is up'",
            "",
        ])

    def build_launch(self, model, settings, session, source_note=""):
        params = self._params_block(settings)
        keep_alive = str(settings.get("keep_alive") or "30m").strip() or "30m"
        presence = str(settings.get("presence_penalty") or "").strip()

        notes = []
        target = model.id
        tag = ""
        if params:
            tag = derived_tag(model.id)
            target = tag
            notes.append(
                "Creates the temporary tag %s (about 1 KB - the weights are "
                "shared with %s, nothing is duplicated). It is deleted when you "
                "unload." % (tag, model.id))
        else:
            notes.append("No settings filled in, so %s is loaded as-is and no "
                         "temporary tag is created." % model.id)
        if presence:
            notes.append("presence_penalty applies to this load only - Ollama "
                         "cannot bake it into a model.")

        script_path, log_path = runner.new_paths(self.name, model.id)
        parts = [self._preamble(log_path, "load model", model.id, source_note),
                 self._ensure_server()]

        if params:
            pairs = "; ".join(
                "%s = %s" % (k, json.dumps(v)) for k, v in sorted(params.items()))
            parts.append("\n".join([
                "# --- temporary tuned tag -------------------------------------",
                "# /api/create reuses the base model's weight layers by digest;",
                "# Ollama reports 'using already created layer sha256:...' as it works.",
                "$params = @{ %s }" % pairs,
                "$createBody = @{",
                "  model      = %s" % runner.ps_single(tag),
                "  from       = %s" % runner.ps_single(model.id),
                "  parameters = $params",
                "  stream     = $false",
                "} | ConvertTo-Json -Depth 6",
                "Write-Log %s" % runner.ps_single("creating " + tag),
                "try {",
                "  $r = Invoke-RestMethod \"$base/api/create\" -Method Post "
                "-ContentType 'application/json' -Body $createBody -TimeoutSec 600",
                "  Write-Log ('create: ' + $r.status)",
                "} catch {",
                "  Write-Log ('create failed: ' + $_.Exception.Message)",
                "  exit 6",
                "}",
                "",
            ]))

        load_opts = ""
        if presence:
            load_opts = "  options    = @{ presence_penalty = %s }\n" % presence
        parts.append("\n".join([
            "# --- load into VRAM ------------------------------------------",
            "# An empty prompt loads the model without starting a chat.",
            "$loadBody = @{",
            "  model      = %s" % runner.ps_single(target),
            "  prompt     = ''",
            "  keep_alive = %s" % runner.ps_single(keep_alive),
            "  stream     = $false",
            load_opts + "} | ConvertTo-Json -Depth 6",
            "Write-Log %s" % runner.ps_single("loading " + target),
            "try {",
            "  Invoke-RestMethod \"$base/api/generate\" -Method Post "
            "-ContentType 'application/json' -Body $loadBody -TimeoutSec 900 | Out-Null",
            "} catch {",
            "  Write-Log ('load failed: ' + $_.Exception.Message)",
            "  exit 1",
            "}",
            "Write-Log 'loaded'",
            "exit 0",
            "",
        ]))

        return runner.LaunchPlan(
            kind="load", backend=self.name,
            script_text="\n".join(parts),
            script_path=script_path, log_path=log_path,
            endpoint=OLLAMA_BASE, host=OLLAMA_HOST, port=OLLAMA_PORT,
            long_lived=False, creates_tag=tag, notes=notes,
        )

    def build_stop(self, loaded, session):
        """Unload, confirm it is gone, then delete the tag if it is ours.

        The suffix is re-checked inside the script as well as in Python. That
        duplication is deliberate: the script is a artifact the user can read
        and keep, and it should be safe on its own terms.
        """
        tag = loaded.id
        notes = []
        remove = ""
        if tag.endswith(state.TAG_SUFFIX):
            ok, reason = tag_is_ours(tag, session)
            if ok:
                remove = tag
                notes.append("Deletes the temporary tag %s after unloading." % tag)
            else:
                notes.append("Will unload but NOT delete %s: %s" % (tag, reason))
        else:
            notes.append("%s was not created by Zoomies, so only the unload "
                         "happens - nothing is deleted." % tag)

        script_path, log_path = runner.new_paths(self.name, "stop-" + tag)
        parts = [self._preamble(log_path, "unload model", tag, ""),
                 "\n".join([
                     "if (-not (Test-Srv)) { Write-Log 'Ollama is not running - nothing to unload'; exit 0 }",
                     "",
                     "# --- unload (keep_alive 0 evicts it immediately) --------------",
                     "$body = @{",
                     "  model      = %s" % runner.ps_single(tag),
                     "  prompt     = ''",
                     "  keep_alive = 0",
                     "  stream     = $false",
                     "} | ConvertTo-Json -Depth 5",
                     "try {",
                     "  Invoke-RestMethod \"$base/api/generate\" -Method Post "
                     "-ContentType 'application/json' -Body $body -TimeoutSec 120 | Out-Null",
                     "  Write-Log 'unload requested'",
                     "} catch { Write-Log ('unload failed: ' + $_.Exception.Message) }",
                     "",
                 ])]

        if remove:
            parts.append("\n".join([
                "# --- wait until it is really gone, then delete the tag --------",
                "$gone = $false",
                "for ($i = 0; $i -lt 30; $i++) {",
                "  Start-Sleep -Milliseconds 400",
                "  try { $ps = Invoke-RestMethod \"$base/api/ps\" -TimeoutSec 5 } catch { continue }",
                "  $names = @($ps.models | ForEach-Object { $_.name })",
                "  if ($names -notcontains %s) { $gone = $true; break }" % runner.ps_single(remove),
                "}",
                "if (-not $gone) { Write-Log 'still loaded - leaving the tag in place'; exit 0 }",
                "",
                "# Safety: only ever delete inside the reserved namespace.",
                "$tag = %s" % runner.ps_single(remove),
                "if (-not $tag.EndsWith(%s)) { Write-Log 'refusing to delete outside the zoomies namespace'; exit 0 }"
                % runner.ps_single(state.TAG_SUFFIX),
                "try {",
                "  Invoke-RestMethod \"$base/api/delete\" -Method Delete "
                "-ContentType 'application/json' -Body (@{ model = $tag } | ConvertTo-Json) "
                "-TimeoutSec 60 | Out-Null",
                "  Write-Log ('deleted ' + $tag)",
                "} catch { Write-Log ('delete failed: ' + $_.Exception.Message) }",
                "",
            ]))
        parts.append("exit 0\n")

        return runner.LaunchPlan(
            kind="stop", backend=self.name,
            script_text="\n".join(parts),
            script_path=script_path, log_path=log_path,
            endpoint=OLLAMA_BASE, host=OLLAMA_HOST, port=OLLAMA_PORT,
            long_lived=False, removes_tag=remove, notes=notes,
        )

    def build_sweep(self, tags):
        """Delete orphaned tags left behind by a crash. Callers must have run
        can_remove_tag on every name first."""
        script_path, log_path = runner.new_paths(self.name, "sweep")
        parts = [self._preamble(log_path, "remove leftover tags",
                                ", ".join(tags), "")]
        parts.append("if (-not (Test-Srv)) { Write-Log 'Ollama is not running'; exit 0 }\n")
        for tag in tags:
            parts.append("\n".join([
                "$tag = %s" % runner.ps_single(tag),
                "if ($tag.EndsWith(%s)) {" % runner.ps_single(state.TAG_SUFFIX),
                "  try {",
                "    Invoke-RestMethod \"$base/api/delete\" -Method Delete "
                "-ContentType 'application/json' -Body (@{ model = $tag } | ConvertTo-Json) "
                "-TimeoutSec 60 | Out-Null",
                "    Write-Log ('deleted ' + $tag)",
                "  } catch { Write-Log ('delete failed: ' + $_.Exception.Message) }",
                "}",
                "",
            ]))
        parts.append("exit 0\n")
        return runner.LaunchPlan(
            kind="stop", backend=self.name,
            script_text="\n".join(parts),
            script_path=script_path, log_path=log_path,
            endpoint=OLLAMA_BASE, notes=[],
        )


register(OllamaBackend())


# --------------------------------------------------------------------------
# llama.cpp
# --------------------------------------------------------------------------
#
# The engine underneath both other backends, run directly. Every setting is
# a command-line flag, so the launch script is one command line and nothing
# is written anywhere else.
#
# Zoomies starts its own `llama.exe serve` per model on its own port rather
# than loading into the llama.cpp app's router: the router takes a model's
# flags from a saved presets file, so per-load settings would mean editing
# that file - a permanent change. Models the app has loaded still show up in
# Loaded, and can be unloaded through the router's own /models/unload.

LLAMACPP_HOST = "127.0.0.1"
LLAMACPP_PORT = 8080            # the first port Zoomies serves on

# Sampling set this way is the server's default: a request that sends its
# own values still wins.
LLAMACPP_FLAGS = (
    ("temperature", "--temp"), ("top_p", "--top-p"), ("top_k", "--top-k"),
    ("min_p", "--min-p"), ("repeat_penalty", "--repeat-penalty"),
    ("presence_penalty", "--presence-penalty"), ("seed", "--seed"),
    ("context_length", "-c"), ("gpu_layers", "-ngl"), ("parallel", "-np"),
)
LLAMACPP_INT_FLAGS = {"top_k", "seed", "context_length", "gpu_layers", "parallel"}
LLAMACPP_SAMPLING = {"temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                     "presence_penalty", "seed"}
# Zoomies decides where the server listens and which model it serves.
LLAMACPP_DENIED = {"--port", "--host", "-m", "--model", "-hf", "-hfr",
                   "--hf-repo", "--models-dir", "--models-preset"}
# Shared-memory GPUs report system RAM as VRAM, so llama.cpp will place
# layers there - and every token then crawls across the bus.
INTEGRATED_GPU = re.compile(
    r"(?i)\b(UHD|Iris|Intel\(R\) Graphics|Radeon\(TM\) Graphics|Radeon Graphics)\b")
HF_REPO_QUANT = re.compile(r"^[\w.\-]+/[\w.\-]+(:[\w.\-]+)?$")


def find_llama_exe():
    """llama.exe from the llama.cpp app (an App Execution Alias under
    WindowsApps), or llama-server.exe from a release zip. Never the copy
    Unsloth Studio bundles, should it still be installed."""
    dirs = (os.environ.get("PATH") or "").split(os.pathsep)
    dirs.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft",
                             "WindowsApps"))
    for name in ("llama.exe", "llama-server.exe"):
        for directory in dirs:
            directory = directory.strip('"')
            if not directory:
                continue
            candidate = os.path.join(directory, name)
            # os.path.exists, not isfile: an execution alias is a reparse point
            if os.path.exists(candidate) and "unsloth" not in candidate.lower():
                return candidate
    return ""


def hf_hub_dir():
    """The Hugging Face cache, where `llama.exe -hf` downloads land."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


_OLLAMA_NAMES = {"at": 0.0, "by_blob": {}}


def ollama_name_for(path):
    """The Ollama tag for one of Ollama's blob files, or "".

    A llama-server started on a blob - which is how this app runs an Ollama
    model under llama.cpp - has nothing to report but the sha256 file name.
    Nobody can read that, and "sha256-f5f1dd89..." in History is no more use
    than a blank. Re-read every half minute: models get pulled while the app
    is open, and this runs on the polling thread.
    """
    name = os.path.basename(str(path or ""))
    if not name.startswith("sha256-"):
        return ""
    now = time.time()
    if now - _OLLAMA_NAMES["at"] > 30:
        _OLLAMA_NAMES["by_blob"] = {os.path.basename(blob): tag
                                    for tag, blob, _size in ollama_manifests()}
        _OLLAMA_NAMES["at"] = now
    return _OLLAMA_NAMES["by_blob"].get(name, "")


def looks_like_path(value):
    """A file, rather than a name a backend takes.

    A bare forward slash cannot be the test: ggml-org/Qwen3.5-0.8B-GGUF:Q8_0
    is what `-hf` is given, and cutting it down to its last segment would
    lose the model.
    """
    value = str(value or "")
    return bool("\\" in value or value.lower().endswith(".gguf")
                or re.match(r"^[A-Za-z]:", value) or value.startswith("/"))


_LIVE_SHELLS = {}


def record_is_live(rec, ttl=5.0):
    """Is the PowerShell that started this server still running?

    What makes a remembered label safe to show. Once that shell is gone the
    server it launched is gone too, and anything answering on that port now is
    something else - so the record names a model that is no longer there.
    Cached briefly: this is asked on the polling thread, and tasklist is not
    free.
    """
    pid = int((rec or {}).get("shell_pid") or 0)
    if not pid:
        return False
    now = time.time()
    seen = _LIVE_SHELLS.get(pid)
    if seen is None or now - seen[0] > ttl:
        seen = (now, state.alive_and_named(pid, "powershell"))
        _LIVE_SHELLS[pid] = seen
    return seen[1]


def props_model_name(props):
    """What a llama-server says it is holding, from GET /props.

    Its --alias when it was given one (Zoomies always passes the name the
    user picked), otherwise the weights file it opened.
    """
    for key in ("model_alias", "model_path"):
        value = str((props or {}).get(key) or "").strip()
        if not value:
            continue
        if not looks_like_path(value):
            return value
        return (ollama_name_for(value)
                or os.path.splitext(os.path.basename(value))[0])
    return ""


def seconds_from(text):
    """"30m" -> 1800, "90" -> 90, "1h" -> 3600; None if unreadable."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*([smh]?)\s*", str(text or "").lower())
    if not m:
        return None
    return int(float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)])


def setting_number(settings, key, as_int=False):
    raw = settings.get(key, "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(str(raw).replace(",", ""))
    except ValueError:
        return None
    return int(value) if as_int else value


def split_extra_flags(settings, denied):
    """Split the free-text flags field; returns (kept, dropped)."""
    raw = str(settings.get("extra_flags") or "").strip()
    if not raw:
        return [], []
    try:
        import shlex
        parts = shlex.split(raw, posix=False)
    except ValueError:
        parts = raw.split()
    kept, dropped, skip_next = [], [], False
    for i, token in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if token.split("=", 1)[0] in denied:
            dropped.append(token)
            if "=" not in token and i + 1 < len(parts) \
                    and not parts[i + 1].startswith("-"):
                skip_next = True
            continue
        kept.append(token)
    return kept, dropped


_DEVICES = {}


def list_devices(exe, prefix):
    """[(id, description)] from --list-devices, asked once per run."""
    if exe in _DEVICES:
        return _DEVICES[exe]
    try:
        res = subprocess.run([exe] + prefix + ["--list-devices"],
                             capture_output=True, text=True, timeout=60,
                             creationflags=state.CREATE_NO_WINDOW)
        text = (res.stdout or "") + (res.stderr or "")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    # "Vulkan1: Intel(R) UHD Graphics 770 (32622 MiB, ...)" - the name itself
    # can hold brackets, so it ends at the memory figure, not the first one.
    found = re.findall(r"^\s*([A-Za-z]+\d+):\s*(.+?)\s*\(\d+ MiB", text, re.M)
    if found:
        _DEVICES[exe] = found
    return found


# The quantisation as it is written in a file name or an Ollama tag.
QUANT_RE = re.compile(r"(?i)(UD-[A-Z0-9_]+|IQ\d[_A-Z0-9]*|"
                      r"Q\d[_A-Z0-9]*|MXFP4|BF16|F16|F32)")


def scan_gguf_folder(backend, folder):
    """Loose .gguf files under a folder.

    Skips the pieces that are not a model you can load on their own:
    projector files, speculative-decoding sidecars, llama.cpp's tiny vocab
    fixtures, and every shard of a split model except the first (llama.cpp
    finds the rest itself).
    """
    if not folder or not os.path.isdir(folder):
        return []
    skip = ("mmproj", "dspark", "ggml-vocab", "-draft")
    out, seen = [], set()
    for base, dirs, files in os.walk(folder):
        if base[len(folder):].count(os.sep) >= 4:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            low = fname.lower()
            if not low.endswith(".gguf") or any(s in low for s in skip):
                continue
            shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", low)
            if shard and shard.group(1) != "00001":
                continue
            path = os.path.join(base, fname)
            if path in seen:
                continue
            record = gguf_record(backend, path)
            if record is not None:
                seen.add(path)
                out.append(record)
    return out


def gguf_record(backend, path):
    """One .gguf file as a model, or None if it is too small to be one."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size < 50 * 1024 * 1024:
        return None
    fname = os.path.basename(path)
    quant = QUANT_RE.search(fname)
    return ModelRecord(
        backend=backend, id=path, label=os.path.splitext(fname)[0],
        quant=quant.group(1) if quant else "",
        size_bytes=size, source="folder", gguf_path=path)


class LlamaCppBackend(Backend):
    name = "llamacpp"
    display_name = "llama.cpp"
    uses_model_folder = True
    host = LLAMACPP_HOST
    port = LLAMACPP_PORT

    def __init__(self):
        Backend.__init__(self)
        self.exe = find_llama_exe()
        # llama.exe is a multi-tool (`llama serve`, `llama download`);
        # llama-server.exe is the server alone.
        self.prefix = ["serve"] if os.path.basename(self.exe).lower() == "llama.exe" else []
        self.default_folder = hf_hub_dir()
        self.endpoint = "http://%s:%d" % (LLAMACPP_HOST, LLAMACPP_PORT)
        self._ports, self._ports_at = {}, 0.0

    # -- availability ------------------------------------------------------

    def is_available(self):
        if not self.exe:
            return False, "llama.cpp not found"
        return True, "installed"

    def can_download(self):
        return bool(self.prefix)

    # -- discovery ---------------------------------------------------------

    def list_models(self, folder=None):
        models = self._from_cache()
        for source in (self._from_ollama(), scan_gguf_folder(self.name, folder)):
            seen = {os.path.normcase(m.gguf_path) for m in models}
            for record in source:
                if os.path.normcase(record.gguf_path) not in seen:
                    models.append(record)
        models.sort(key=lambda m: m.label.lower())
        return models

    def _from_ollama(self):
        """Models Ollama has already pulled, run straight from its blobs.

        Both backends are llama.cpp reading the same GGUF, and on a machine
        that uses Ollama its store is where the models actually are. Without
        this the dropdown offers only what the Hugging Face cache happens to
        hold - usually just the one file "Download..." fetched - and Rescan
        looks broken, because there is never anything else to find. Nothing
        is copied or converted: the blob goes to -m as it is.

        The Ollama name is kept as the id, so it becomes --alias and one model
        reads the same in Loaded and History whichever backend ran it, which
        is the whole point of comparing the two.
        """
        out = []
        for name, blob, size in ollama_manifests():
            if not is_gguf(blob):
                continue
            quant = QUANT_RE.search(name)
            out.append(ModelRecord(
                backend=self.name, id=name, label=name,
                quant=quant.group(1) if quant else "",
                size_bytes=size, source="ollama", gguf_path=blob))
        return out

    def _from_cache(self):
        """Models in the Hugging Face cache, named the way `-hf` takes them:
        ggml-org/Qwen3.5-0.8B-GGUF:Q8_0."""
        hub = hf_hub_dir()
        try:
            entries = os.listdir(hub)
        except OSError:
            return []
        found = {}
        for entry in entries:
            if not entry.startswith("models--"):
                continue
            repo = entry[len("models--"):].replace("--", "/", 1)
            snaps = os.path.join(hub, entry, "snapshots")
            try:
                revs = sorted((os.path.getmtime(os.path.join(snaps, r)), r)
                              for r in os.listdir(snaps))
            except OSError:
                continue
            for _mtime, rev in reversed(revs):          # newest revision wins
                for rec in scan_gguf_folder(self.name, os.path.join(snaps, rev)):
                    if rec.quant:
                        mid = "%s:%s" % (repo, rec.quant)
                        source = "hf-cache"
                    else:
                        mid, source = rec.gguf_path, "folder"
                    if mid in found:
                        continue
                    found[mid] = ModelRecord(
                        backend=self.name, id=mid, label=mid, quant=rec.quant,
                        size_bytes=rec.size_bytes, source=source,
                        gguf_path=rec.gguf_path)
        # Files saved straight into the hub folder rather than through -hf -
        # a browser download, say - sit loose beside the models--* entries.
        # Only the top level: everything below it is the cache's own layout,
        # handled above.
        for entry in entries:
            low = entry.lower()
            if not low.endswith(".gguf") or "mmproj" in low:
                continue
            rec = gguf_record(self.name, os.path.join(hub, entry))
            if rec is not None:
                found.setdefault(rec.gguf_path, rec)
        return list(found.values())

    # -- capability matrix -------------------------------------------------

    def supports(self, key):
        if key in LLAMACPP_SAMPLING:
            return "server default - a request's own value wins"
        return {
            "context_length": "-c",
            "gpu_layers": "-ngl",
            "parallel": "-np (slots share the context)",
            "keep_alive": "sleeps when idle, wakes on the next request",
            "extra_flags": "passed to llama-server",
            "reasoning": "applied at launch",
            "kv_cache": "applied at launch",
        }.get(key, "")

    # -- what is loaded ----------------------------------------------------

    def _servers(self, session=None):
        session = session if session is not None else state.load_session()
        return (session.get(self.name) or {}).get("servers") or {}

    def list_loaded(self):
        """Every llama.cpp server on this machine, named by the server itself.

        The name comes from /props, not from the record written when Zoomies
        started it. Ports get reused - the next load takes the first free one
        from 8080, and a run of benchmarks restarts servers all day - so the
        remembered label is a model behind the moment that happens, and every
        request then gets filed under the wrong name. The server always knows
        what it actually loaded; the record is only a fallback for when it is
        too busy to answer, and for telling ours from everyone else's.

        Servers started outside Zoomies are listed too: one holding a model in
        VRAM is exactly what the user needs to see, whoever started it.
        """
        records = {}
        for rec in self._servers().values():
            port = int(rec.get("port") or 0)
            if port:
                records[port] = rec
        found = self._ports_now()
        ports = sorted(set(records) | {p for p, (be, _pid) in found.items()
                                       if be == self.name})

        servers, routers = [], []
        for port in ports:
            rec = records.get(port)
            if not state.port_open(LLAMACPP_HOST, port):
                continue
            base = "http://%s:%d" % (LLAMACPP_HOST, port)
            props, err = http_json(base + "/props", timeout=3.0)
            # llama-server answers /props with 503 while a slot is busy
            # generating. That is not "unloaded": the port is open and the
            # model is in VRAM. Dropping it here made the Loaded table empty
            # out mid-request and History fall back to whatever the dropdown
            # showed, filing a whole benchmark run under the wrong model.
            if err and "503" not in err:
                continue
            if props is None:
                # Busy, so the only name available is the remembered one, and
                # that is worth having only if it is this server's. A record
                # whose launcher has exited belongs to a server that is gone,
                # and the port has since been handed to somebody else.
                if rec is None or not record_is_live(rec):
                    continue
            if (props or {}).get("role") == "router":
                routers.append((base, found.get(port, ("", 0))[1]))
            else:
                servers.append((port, base, props, rec))

        out, app_models = [], set()
        for base, pid in routers:
            for item in self._router_models(base, pid):
                app_models.add(item.label)
                out.append(item)
        for port, base, props, rec in servers:
            name = props_model_name(props)
            label = name or (rec or {}).get("label") or (rec or {}).get("model_id", "")
            # The id is what a backend is told to act on, so a remembered one
            # is worth more than a name read back off a file path.
            model_id = (rec or {}).get("model_id") or name or label
            # The router runs each model in a child server on a port of its
            # own. The router has already reported those, and it is the router
            # that can unload them, so a child is not listed a second time.
            if rec is None and label in app_models:
                continue
            settings = (props or {}).get("default_generation_settings") or {}
            out.append(LoadedModel(
                backend=self.name, id=model_id, label=label,
                context=int(settings.get("n_ctx") or 0),
                endpoint=base,
                pid=(int((rec or {}).get("shell_pid") or 0)
                     or found.get(port, ("", 0))[1]),
                owned_by_us=rec is not None,
                foreign_process=rec is None,
                log_path=(rec or {}).get("log", "")))
        return out

    def _ports_now(self):
        """{port: (backend, pid)} for every llama-server, asked every 5 s."""
        import metrics                   # port discovery lives there
        now = time.time()
        if now - self._ports_at > 5:
            self._ports, self._ports_at = metrics.llama_server_ports(), now
        return self._ports

    def model_on_port(self, port):
        """The model on one port, asked directly - for naming a request that
        arrived before the two-second poll had seen the server at all."""
        if not port:
            return ""
        props, _err = http_json("http://%s:%d/props" % (LLAMACPP_HOST, int(port)),
                                timeout=1.0)
        return props_model_name(props)

    @staticmethod
    def _router_models(base, pid):
        """Models the llama.cpp app's router has loaded."""
        listing, _err = http_json(base + "/v1/models", timeout=3.0)
        out = []
        for item in (listing or {}).get("data", []) or []:
            if ((item.get("status") or {}).get("value")) != "loaded":
                continue
            name = item.get("id", "")
            out.append(LoadedModel(
                backend="llamacpp", id=name, label=name, endpoint=base,
                pid=pid, owned_by_us=False))
        return out

    def remember_server(self, session, plan, shell_pid, model=None):
        # From the plan, not from whatever the dropdown shows by the time the
        # server finishes coming up: a big model takes half a minute to load,
        # and the selection can have moved on - after a Rescan, say - which
        # left the record naming a model this server never held.
        servers = session.setdefault(self.name, {}).setdefault("servers", {})
        servers[str(plan.port)] = {
            "shell_pid": shell_pid, "port": plan.port,
            "model_id": plan.model_id,
            "label": plan.model_label or plan.model_id,
            "log": plan.log_path, "script": plan.script_path,
            "started": time.time(),
        }

    def forget_server(self, session, plan):
        if plan.port:
            self._servers(session).pop(str(plan.port), None)

    # -- script generation -------------------------------------------------

    def _preamble(self, log_path, title, model, source_note):
        return "\n".join([
            runner.header(title, model, source_note),
            "",
            "$ErrorActionPreference = 'Stop'",
            "$log  = %s" % runner.ps_single(log_path),
            "",
            "function Write-Log {",
            "  param($m)",
            "  $line = '[zoomies] ' + (Get-Date).ToString('o') + ' ' + $m",
            "  Write-Output $line",
            "  $line | Out-File -FilePath $log -Append -Encoding utf8",
            "}",
            "",
        ])

    def _native_call(self, args, what):
        """Run llama.exe with its output in the log.

        Windows PowerShell 5.1 turns each stderr line of a native program
        into an error record, and llama.cpp logs everything to stderr: under
        'Stop' the first line would end the script and the server with it.
        """
        return "\n".join([
            "$exe = %s" % runner.ps_single(self.exe),
            "$a = @(",
            "\n".join("  " + a for a in args),
            ")",
            "Write-Log ('%s: ' + ($a -join ' '))" % what,
            "$ErrorActionPreference = 'Continue'",
            "& $exe @a 2>&1 | ForEach-Object { \"$_\" | Out-File -FilePath $log -Append -Encoding utf8 }",
            "$code = $LASTEXITCODE",
            "Write-Log ('llama.cpp exited with ' + $code)",
            "exit $code",
            "",
        ])

    def free_port(self, session):
        # Only ports actually in use count: a record left by a server that
        # died, or was ended from Processes, must not push every later load
        # one port further up.
        for port in range(LLAMACPP_PORT, LLAMACPP_PORT + 50):
            if not state.port_open(LLAMACPP_HOST, port):
                return port
        return LLAMACPP_PORT

    def build_launch(self, model, settings, session, source_note=""):
        port = self.free_port(session)
        base = "http://%s:%d" % (LLAMACPP_HOST, port)
        script_path, log_path = runner.new_paths(self.name, model.label)
        extra, dropped = split_extra_flags(settings, LLAMACPP_DENIED)
        notes = ["Starts llama.cpp on port %d. Apps connect to %s/v1; its "
                 "own chat page is at %s." % (port, base, base)]
        if dropped:
            notes.append("Ignoring %s - Zoomies sets the model and port itself."
                         % ", ".join(dropped))

        a = [runner.ps_single(p) for p in self.prefix]
        if model.source == "hf-cache":
            # -hf finds the model in the cache along with its vision projector
            # and any split parts; --offline keeps it from touching the network.
            a += ["'-hf'", runner.ps_single(model.id), "'--offline'"]
        else:
            a += ["'-m'", runner.ps_single(model.gguf_path or model.id)]
        a += ["'--alias'", runner.ps_single(model.id)]
        for key, flag in LLAMACPP_FLAGS:
            value = setting_number(settings, key, key in LLAMACPP_INT_FLAGS)
            if value is not None:
                a += [runner.ps_single(flag), runner.ps_single(value)]
        idle = seconds_from(settings.get("keep_alive"))
        if idle and idle > 0:
            a += ["'--sleep-idle-seconds'", runner.ps_single(idle)]
        kv_type = kv_choice_value(settings.get("kv_cache"))
        if kv_type:
            a += ["'-ctk'", runner.ps_single(kv_type)]
            if kv_type in KV_QUANTIZED and flash_attn_off(extra):
                notes.append("KV cache: K at %s, V left at f16 - a quantized V "
                             "cache needs flash attention, and Extra flags turn "
                             "it off (-fa off)." % kv_type)
            else:
                a += ["'-ctv'", runner.ps_single(kv_type)]
        kwargs = reasoning_kwargs(settings)
        if kwargs:
            # PowerShell 5.1 strips bare double quotes from native arguments;
            # backslash-escaped ones arrive intact.
            raw = json.dumps(kwargs, separators=(",", ":"))
            a += ["'--chat-template-kwargs'", runner.ps_single(raw.replace('"', '\\"'))]
        if not any(t.split("=", 1)[0] in ("-dev", "--device") for t in extra):
            devices = list_devices(self.exe, self.prefix)
            dedicated = [d for d in devices if not INTEGRATED_GPU.search(d[1])]
            if dedicated and len(dedicated) < len(devices):
                a += ["'--device'", runner.ps_single(",".join(d[0] for d in dedicated))]
                notes.append("Using %s. Skipping %s: it shares system memory, "
                             "so layers placed there run slowly."
                             % (", ".join(d[1] for d in dedicated),
                                ", ".join(d[1] for d in devices if d not in dedicated)))
        a += [runner.ps_single(t) for t in extra]
        a += ["'--host'", runner.ps_single(LLAMACPP_HOST),
              "'--port'", runner.ps_single(port), "'--log-colors'", "'off'"]

        if not self.exe:
            body = "\n".join([self._preamble(log_path, "start llama.cpp", model.label,
                                             source_note),
                              "Write-Log 'llama.cpp was not found'", "exit 2", ""])
        else:
            body = "\n".join([self._preamble(log_path, "start llama.cpp", model.label,
                                             source_note),
                              self._native_call(a, "starting")])
        return runner.LaunchPlan(
            kind="load", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path, endpoint=base,
            host=LLAMACPP_HOST, port=port, long_lived=True, notes=notes,
            model_id=model.id, model_label=model.label,
            ready_check=lambda: self._ready(port))

    def _ready(self, port):
        data, err = http_json("http://%s:%d/health" % (LLAMACPP_HOST, port),
                              timeout=3.0)
        return not err and (data or {}).get("status") == "ok"

    def build_stop(self, loaded, session):
        port = 0
        m = re.search(r":(\d+)$", loaded.endpoint or "")
        if m:
            port = int(m.group(1))
        record = self._servers(session).get(str(port)) if port else None
        script_path, log_path = runner.new_paths(self.name, "stop-" + loaded.label)
        # Only the llama.cpp app's router can unload a model while staying up.
        # A plain server - ours, or one somebody started by hand - is stopped
        # by ending the process holding its port, which is also the only thing
        # that gives the VRAM back. A server that does not answer is left to
        # the unload request: it may be a router busy with a prompt, and
        # killing the app when it was only slow to reply is unforgivable.
        props, err = http_json((loaded.endpoint or "") + "/props", timeout=2.0)
        plain = not err and (props or {}).get("role") != "router"

        if record or (port and plain):
            shell_pid = int((record or {}).get("shell_pid") or 0)
            notes = ["Stops the llama.cpp server on port %d." % port]
            if not record:
                notes.append("This server was not started by Zoomies.")
            body = "\n".join([
                self._preamble(log_path, "stop llama.cpp", loaded.label, ""),
                "$ErrorActionPreference = 'Continue'",
                "$port = %d" % port,
                "$pids = @(%d)" % shell_pid,
                "$pids += @(Get-NetTCPConnection -LocalPort $port -State Listen "
                "-ErrorAction SilentlyContinue | ForEach-Object { $_.OwningProcess })",
                "foreach ($p in ($pids | Where-Object { $_ } | Select-Object -Unique)) {",
                "  $proc = Get-Process -Id $p -ErrorAction SilentlyContinue",
                "  # A remembered pid can be reused by an unrelated program after a",
                "  # restart; only ever kill something that is plausibly ours.",
                "  if ($proc -and $proc.ProcessName -in @('powershell','llama','llama-server')) {",
                "    taskkill /T /F /PID $p 2>&1 | ForEach-Object { \"$_\" | Out-File -FilePath $log -Append -Encoding utf8 }",
                "  }",
                "}",
                "$n = 0",
                "while ((Get-NetTCPConnection -LocalPort $port -State Listen "
                "-ErrorAction SilentlyContinue) -and $n -lt 40) { Start-Sleep -Milliseconds 250; $n++ }",
                "if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {",
                "  Write-Log 'port is still in use'; exit 1",
                "}",
                "Write-Log 'stopped'",
                "exit 0",
                "",
            ])
        else:
            notes = ["Unloads it from the llama.cpp app, which keeps running."]
            body = "\n".join([
                self._preamble(log_path, "unload from llama.cpp app", loaded.label, ""),
                "$body = @{ model = %s } | ConvertTo-Json" % runner.ps_single(loaded.id),
                "try {",
                "  Invoke-RestMethod %s -Method Post -ContentType 'application/json' "
                "-Body $body -TimeoutSec 120 | Out-Null"
                % runner.ps_single((loaded.endpoint or "") + "/models/unload"),
                "  Write-Log 'unloaded'",
                "} catch { Write-Log ('unload failed: ' + $_.Exception.Message); exit 1 }",
                "exit 0",
                "",
            ])
        return runner.LaunchPlan(
            kind="stop", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            endpoint=loaded.endpoint, host=LLAMACPP_HOST,
            # Carries the port whenever this script ends the server on it, so
            # any record for that port is dropped afterwards rather than left
            # to name whatever starts there next.
            port=port if (record or plain) else 0, notes=notes)

    def build_download(self, repo_quant):
        """`llama download -hf org/repo:QUANT` into the shared cache."""
        script_path, log_path = runner.new_paths(self.name, "download-" + repo_quant)
        a = ["'download'", "'-hf'", runner.ps_single(repo_quant)]
        body = "\n".join([self._preamble(log_path, "download", repo_quant, ""),
                          self._native_call(a, "downloading")])
        return runner.LaunchPlan(
            kind="download", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            notes=["Downloads %s into the Hugging Face cache." % repo_quant])


register(LlamaCppBackend())
