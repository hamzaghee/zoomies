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
        return ""       # gpu_layers, parallel, flash_attn, extra_flags

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
