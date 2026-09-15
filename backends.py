r"""
Zoomies - the backend seam.

This is the one module allowed to know what "Ollama" or "Unsloth" means.
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

# KV cache types, in the order Unsloth Studio's own dropdown lists them, with
# llama.cpp's default spelled out first. One value sets both K and V, the same
# single knob Unsloth exposes.
KV_CACHE_DEFAULT = "f16 (default)"
KV_CACHE_CHOICES = (KV_CACHE_DEFAULT, "bf16", "q8_0", "q4_0", "q4_1",
                    "q5_0", "q5_1", "iq4_nl", "f32")


def kv_choice_value(choice):
    """The cache type to send, or None for llama.cpp's own default."""
    value = str(choice or "").strip()
    if not value or value == KV_CACHE_DEFAULT or value not in KV_CACHE_CHOICES:
        return None
    return value


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
            # Unsloth Studio disables these models the same way.
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

    def fixed_kv_cache(self):
        """A KV cache type this backend imposes regardless of Zoomies, or None."""
        return None


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

        Reads manifest filenames off disk. Deliberately does NOT shell out to
        `ollama list` to recover metadata: that would launch the tray app.
        """
        root = os.path.join(ollama_models_dir(), "manifests")
        out = []
        for base, _dirs, files in os.walk(root):
            for fname in files:
                rel = os.path.relpath(os.path.join(base, fname), root)
                parts = rel.replace("\\", "/").split("/")
                if len(parts) < 2:
                    continue
                name = "%s:%s" % (parts[-2], parts[-1])
                size = 0
                try:
                    with open(os.path.join(base, fname), "r", encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    for layer in manifest.get("layers", []):
                        if str(layer.get("mediaType", "")).endswith(".model"):
                            size = int(layer.get("size") or 0)
                except (OSError, ValueError):
                    pass
                out.append(ModelRecord(
                    backend=self.name, id=name, label=name,
                    size_bytes=size, source="registry",
                ))
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
# Unsloth Studio
# --------------------------------------------------------------------------

UNSLOTH_HOST = "127.0.0.1"
UNSLOTH_PORT = 8888
UNSLOTH_BASE = "http://%s:%d" % (UNSLOTH_HOST, UNSLOTH_PORT)

# Flags Unsloth manages itself and rejects with HTTP 400 if they are passed
# through to llama-server. Its own --host/--port options are used instead.
UNSLOTH_DENIED = (
    "--model", "-m", "-hf", "--hf-repo", "--host", "--port", "--path",
    "--api-prefix", "--reuse-port", "--api-key", "--ssl-cert-file",
    "--ssl-key-file", "--ui", "--webui", "--models-json", "--parallel", "-np",
)

# Settings that map to a first-class `unsloth run` flag.
UNSLOTH_FLAGS = (
    ("temperature", "--temperature"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
    ("seed", "--seed"),
    ("context_length", "--max-seq-length"),
)
UNSLOTH_INT_FLAGS = {"top_k", "seed", "context_length"}

# Settings the running server accepts on POST /v1/load. Note what is absent:
# there is no temperature here. Sampling can only be pinned by the CLI at
# launch, which is why starting the server ourselves is the better path.
UNSLOTH_LOAD_FIELDS = (
    ("context_length", "max_seq_length"),
    ("gpu_layers", "gpu_layers"),
    ("parallel", "n_parallel"),
)


def find_unsloth_exe():
    home = os.path.join(os.path.expanduser("~"), ".unsloth", "studio", "bin",
                        "unsloth.exe")
    if os.path.isfile(home):
        return home
    for directory in (os.environ.get("PATH") or "").split(os.pathsep):
        candidate = os.path.join(directory.strip('"'), "unsloth.exe")
        if os.path.isfile(candidate):
            return candidate
    return ""


class UnslothBackend(Backend):
    """Unsloth Studio.

    Like Ollama, this is driven over HTTP rather than by scripting its CLI -
    the Studio exposes /v1/load, /v1/unload, /v1/status and /v1/models.

    The one thing the API cannot do is pin sampling. POST /v1/load takes
    placement and sizing (max_seq_length, gpu_layers, n_parallel) but has no
    temperature field; `unsloth run --temperature` pins it "for every
    request" instead. So there are two launch shapes:

      server not running -> start it with `unsloth run`, all settings applied
      server running     -> POST /v1/load, and say plainly that the sampling
                            settings cannot be pinned into an already-running
                            server

    That distinction is surfaced through supports() and through a note on the
    plan, rather than being quietly ignored.
    """

    name = "unsloth"
    display_name = "Unsloth Studio"
    uses_model_folder = True
    endpoint = UNSLOTH_BASE
    host = UNSLOTH_HOST
    port = UNSLOTH_PORT

    def __init__(self):
        Backend.__init__(self)
        self.exe = find_unsloth_exe()
        self.default_folder = os.path.join(os.path.expanduser("~"), ".unsloth")

    # -- availability ------------------------------------------------------

    def is_available(self):
        if not self.exe:
            return False, "unsloth.exe not found"
        if self.server_up():
            return True, "running"
        return True, "installed, not running"

    def server_up(self):
        return state.port_open(UNSLOTH_HOST, UNSLOTH_PORT)

    # -- discovery ---------------------------------------------------------

    def list_models(self, folder=None):
        models = self._from_api() if self.server_up() else []
        seen = {m.id for m in models}
        for record in self._from_folder(folder):
            if record.id not in seen:
                models.append(record)
        models.sort(key=lambda m: m.label.lower())
        return models

    def _from_api(self):
        """The running Studio already knows its models, their quants and
        their sizes - better than guessing from filenames."""
        listing, err = http_json(UNSLOTH_BASE + "/v1/models", timeout=8.0)
        if err:
            self.last_error = err
            return []
        self.last_error = ""
        sizes = {}
        cached, cache_err = http_json(UNSLOTH_BASE + "/api/models/cached-gguf",
                                      timeout=10.0)
        if not cache_err:
            for item in (cached or {}).get("cached", []) or []:
                if item.get("repo_id"):
                    sizes[item["repo_id"]] = int(item.get("size_bytes") or 0)
        out = []
        for item in (listing or {}).get("data", []) or []:
            repo = item.get("id") or ""
            if not repo:
                continue
            out.append(ModelRecord(
                backend=self.name, id=repo,
                label=item.get("display_name") or repo,
                quant=item.get("quant", ""),
                size_bytes=sizes.get(repo, 0),
                source="registry"))
        return out

    def _from_folder(self, folder):
        """Loose .gguf files the user points us at.

        Skips the pieces that are not a model you can load on their own:
        projector files, speculative-decoding sidecars, llama.cpp's tiny
        vocab fixtures, and every shard of a split model except the first
        (llama.cpp finds the rest itself).
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
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size < 50 * 1024 * 1024 or path in seen:
                    continue
                seen.add(path)
                quant = re.search(r"(?i)(UD-[A-Z0-9_]+|IQ\d[_A-Z0-9]*|"
                                  r"Q\d[_A-Z0-9]*|BF16|F16)", fname)
                out.append(ModelRecord(
                    backend=self.name, id=path,
                    label=os.path.splitext(fname)[0],
                    quant=quant.group(1) if quant else "",
                    size_bytes=size, source="folder", gguf_path=path))
        return out

    # -- capability matrix -------------------------------------------------

    def supports(self, key):
        if key in ("temperature", "top_p", "top_k", "min_p", "seed"):
            if self.server_up():
                return "needs Zoomies to start the server"
            return "pinned at launch"
        if key == "context_length":
            return "--max-seq-length"
        if key == "gpu_layers":
            return "-ngl"
        if key == "parallel":
            return "decode slots"
        if key in ("repeat_penalty", "presence_penalty"):
            return "passed to llama-server"
        if key == "extra_flags":
            return "passed to llama-server"
        if key == "reasoning":
            return "applied at load"
        if key == "kv_cache":
            return "applied at load"
        return ""          # keep_alive: the model lives as long as the server

    # -- what is loaded ----------------------------------------------------

    def list_loaded(self):
        if not self.server_up():
            return []
        listing, err = http_json(UNSLOTH_BASE + "/v1/models", timeout=5.0)
        if err:
            self.last_error = err
            return []
        self.last_error = ""
        status, _ = http_json(UNSLOTH_BASE + "/v1/status", timeout=5.0)
        context = int((status or {}).get("context_length") or 0)
        session = state.load_session()
        record = session.get("unsloth") or {}
        # Loaded by Zoomies either by starting the server, or by loading into
        # one that was already running. Missing the second case made Unload
        # warn "not started by Zoomies" about a model Zoomies had just loaded.
        ours = {record.get("model_id", ""),
                (session.get("zoomies_loads") or {}).get(self.name, "")} - {""}
        out = []
        for item in (listing or {}).get("data", []) or []:
            if not item.get("loaded"):
                continue
            repo = item.get("id") or ""
            out.append(LoadedModel(
                backend=self.name, id=repo,
                label=item.get("display_name") or repo,
                context=context, endpoint=UNSLOTH_BASE,
                pid=int(record.get("pid") or 0),
                owned_by_us=(repo in ours),
                log_path=record.get("log", "")))
        return out

    def installed_names(self):
        return [m.id for m in self.list_models(None)]

    # -- script generation -------------------------------------------------

    def _preamble(self, log_path, title, model, source_note):
        return "\n".join([
            runner.header(title, model, source_note),
            "",
            "$ErrorActionPreference = 'Stop'",
            "$log  = %s" % runner.ps_single(log_path),
            "$base = %s" % runner.ps_single(UNSLOTH_BASE),
            "",
            "function Write-Log {",
            "  param($m)",
            "  $line = '[zoomies] ' + (Get-Date).ToString('o') + ' ' + $m",
            "  Write-Output $line",
            "  $line | Out-File -FilePath $log -Append -Encoding utf8",
            "}",
            "function Test-Srv {",
            "  try { Invoke-RestMethod \"$base/v1/models\" -TimeoutSec 3 | Out-Null; return $true }",
            "  catch { return $false }",
            "}",
            "",
        ])

    @staticmethod
    def _number(settings, key, as_int=False):
        raw = settings.get(key, "")
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = float(str(raw).replace(",", ""))
        except ValueError:
            return None
        return int(value) if as_int else value

    def _extra_args(self, settings):
        """Split the free-text field, dropping anything Unsloth will reject."""
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
            if token.split("=", 1)[0] in UNSLOTH_DENIED:
                dropped.append(token)
                if "=" not in token and i + 1 < len(parts) \
                        and not parts[i + 1].startswith("-"):
                    skip_next = True
                continue
            kept.append(token)
        return kept, dropped

    def build_launch(self, model, settings, session, source_note=""):
        extra, dropped = self._extra_args(settings)
        notes = []
        if dropped:
            notes.append("Ignoring %s - Unsloth manages those itself."
                         % ", ".join(dropped))
        script_path, log_path = runner.new_paths(self.name, model.label)
        if self.server_up():
            return self._load_into_running(model, settings, extra, notes,
                                           script_path, log_path, source_note)
        return self._start_server(model, settings, extra, notes,
                                  script_path, log_path, source_note)

    def _start_server(self, model, settings, extra, notes, script_path,
                      log_path, source_note):
        """`unsloth run` starts the server and loads the model in one go,
        with the sampling settings pinned for every request."""
        args = ["'run'", "'--model',%s" % runner.ps_single(model.id)]
        for key, flag in UNSLOTH_FLAGS:
            value = self._number(settings, key, key in UNSLOTH_INT_FLAGS)
            if value is not None:
                args.append("'%s',%s" % (flag, runner.ps_single(value)))
        gpu_layers = self._number(settings, "gpu_layers", True)
        if gpu_layers is not None:
            args.append("'-ngl',%s" % runner.ps_single(gpu_layers))
        parallel = self._number(settings, "parallel", True) or 1
        args.append("'--parallel',%s" % runner.ps_single(parallel))
        if model.quant:
            args.append("'--gguf-variant',%s" % runner.ps_single(model.quant))
        args.extend(runner.ps_single(token) for token in extra)
        kwargs = reasoning_kwargs(settings)
        if kwargs:
            # Windows PowerShell 5.1 strips bare double quotes from arguments
            # it hands to a native program, so {"reasoning_effort":"low"}
            # arrives as {reasoning_effort:low} and llama-server rejects it.
            # Backslash-escaped quotes survive the trip - verified by echoing
            # the argument back from a real process.
            raw = json.dumps(kwargs, separators=(",", ":"))
            args.append("'--chat-template-kwargs',%s"
                        % runner.ps_single(raw.replace('"', '\\"')))
        kv_type = kv_choice_value(settings.get("kv_cache"))
        if kv_type:
            # Plain passthrough: Unsloth leaves --cache-type-* to llama.cpp and
            # appends them after its own flags, so these win.
            args.append("'--cache-type-k',%s" % runner.ps_single(kv_type))
            args.append("'--cache-type-v',%s" % runner.ps_single(kv_type))
        # Always explicit, always headless: the requirement is that the
        # backend's own window never opens.
        args.extend(["'--host',%s" % runner.ps_single(UNSLOTH_HOST),
                     "'--port',%s" % runner.ps_single(UNSLOTH_PORT),
                     "'--api-only'"])

        notes.insert(0, "Starts Unsloth Studio on port %d with these settings "
                        "pinned. No Unsloth window opens." % UNSLOTH_PORT)
        body = "\n".join([
            self._preamble(log_path, "start Unsloth and load", model.label,
                           source_note),
            "$exe = %s" % runner.ps_single(self.exe),
            "$a = @(",
            "\n".join("  " + a for a in args),
            ")",
            "Write-Log ('starting: ' + ($a -join ' '))",
            "& $exe @a *>> $log",
            "Write-Log ('unsloth exited with ' + $LASTEXITCODE)",
            "",
        ])
        return runner.LaunchPlan(
            kind="load", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            endpoint=UNSLOTH_BASE, host=UNSLOTH_HOST, port=UNSLOTH_PORT,
            long_lived=True, notes=notes, model_id=model.id)

    def _load_into_running(self, model, settings, extra, notes, script_path,
                           log_path, source_note):
        """POST /v1/load against a server that is already up."""
        fields = {}
        for key, target in UNSLOTH_LOAD_FIELDS:
            value = self._number(settings, key, True)
            if value is not None:
                fields[target] = value
        if model.quant:
            fields["gguf_variant"] = model.quant
        kv_type = kv_choice_value(settings.get("kv_cache"))
        if kv_type:
            fields["cache_type_kv"] = kv_type

        pinned = [label for key, label in
                  (("temperature", "Temperature"), ("top_p", "Top P"),
                   ("top_k", "Top K"), ("min_p", "Min P"), ("seed", "Seed"))
                  if str(settings.get(key) or "").strip()]
        if pinned:
            notes.append(
                "Unsloth Studio is already running, and its load API has no "
                "sampling fields - %s will NOT be applied. Stop the server "
                "first if you need those pinned." % ", ".join(pinned))

        lines = ["$body = @{",
                 "  model_path   = %s" % runner.ps_single(model.id),
                 "  force_reload = $true"]
        for key in sorted(fields):
            value = fields[key]
            lines.append("  %-12s = %s" % (
                key, runner.ps_single(value) if isinstance(value, str) else value))
        passthrough = list(extra)
        kwargs = reasoning_kwargs(settings)
        if kwargs:
            # Sent inside a JSON body, not on a command line, so no quote
            # escaping here - ConvertTo-Json takes care of it.
            passthrough += ["--chat-template-kwargs",
                            json.dumps(kwargs, separators=(",", ":"))]
        if passthrough:
            lines.append("  llama_extra_args = @(%s)"
                         % ", ".join(runner.ps_single(t) for t in passthrough))
        lines.append("} | ConvertTo-Json -Depth 6")

        body = "\n".join([
            self._preamble(log_path, "load model", model.label, source_note),
            "if (-not (Test-Srv)) { Write-Log 'Unsloth is not responding'; exit 3 }",
            "",
            "\n".join(lines),
            "Write-Log %s" % runner.ps_single("loading " + model.id),
            "try {",
            "  Invoke-RestMethod \"$base/v1/load\" -Method Post "
            "-ContentType 'application/json' -Body $body -TimeoutSec 1800 | Out-Null",
            "} catch {",
            "  Write-Log ('load failed: ' + $_.Exception.Message)",
            "  exit 1",
            "}",
            "Write-Log 'loaded'",
            "exit 0",
            "",
        ])
        return runner.LaunchPlan(
            kind="load", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            endpoint=UNSLOTH_BASE, host=UNSLOTH_HOST, port=UNSLOTH_PORT,
            long_lived=False, notes=notes, model_id=model.id)

    def build_stop(self, loaded, session):
        script_path, log_path = runner.new_paths(self.name,
                                                 "stop-" + loaded.label)
        notes = ["Unloads the model. Unsloth Studio itself keeps running."]
        if not loaded.owned_by_us:
            notes.append("This model was loaded outside Zoomies.")
        body = "\n".join([
            self._preamble(log_path, "unload model", loaded.label, ""),
            "if (-not (Test-Srv)) { Write-Log 'Unsloth is not running'; exit 0 }",
            "",
            "$body = @{ model_path = %s } | ConvertTo-Json"
            % runner.ps_single(loaded.id),
            "try {",
            "  Invoke-RestMethod \"$base/v1/unload\" -Method Post "
            "-ContentType 'application/json' -Body $body -TimeoutSec 300 | Out-Null",
            "  Write-Log 'unloaded'",
            "} catch { Write-Log ('unload failed: ' + $_.Exception.Message) }",
            "exit 0",
            "",
        ])
        return runner.LaunchPlan(
            kind="stop", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            endpoint=UNSLOTH_BASE, host=UNSLOTH_HOST, port=UNSLOTH_PORT,
            notes=notes)

    def build_shutdown(self, session):
        """Stop the whole server. Only offered when Zoomies started it -
        `unsloth studio stop` is all-or-nothing for a STUDIO_HOME and would
        take down servers somebody else launched."""
        record = session.get("unsloth") or {}
        pid = int(record.get("pid") or 0)
        script_path, log_path = runner.new_paths(self.name, "shutdown")
        kill_parent = ""
        if pid:
            # unsloth.exe spawns llama-server.exe; killing only the parent
            # leaves the child holding every byte of VRAM.
            kill_parent = ("if (Get-Process -Id %d -ErrorAction SilentlyContinue) "
                           "{ taskkill /T /F /PID %d *>> $log }" % (pid, pid))
        body = "\n".join([
            self._preamble(log_path, "stop Unsloth Studio", "", ""),
            "& %s studio stop *>> $log" % runner.ps_single(self.exe),
            "Start-Sleep -Milliseconds 2000",
            kill_parent,
            "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
            "-ErrorAction SilentlyContinue | ForEach-Object "
            "{ taskkill /T /F /PID $_.ProcessId *>> $log }",
            "Write-Log 'stopped'",
            "exit 0",
            "",
        ])
        return runner.LaunchPlan(
            kind="stop", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            endpoint=UNSLOTH_BASE, notes=["Stops Unsloth Studio entirely."])


register(UnslothBackend())
