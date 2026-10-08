r"""
Zoomies - the stable-diffusion.cpp backend: image generation, not chat.

Why this is a separate backend and not a llama.cpp preset
---------------------------------------------------------
A diffusion model is a GGUF, but GGUF is only a container. llama.cpp
implements transformer *language* models, plus vision encoders used as
projectors for image INPUT. Qwen-Image is a diffusion transformer: iterative
denoising followed by a VAE decode. llama.cpp has no implementation of that
graph and no binary that does it - its shipped executables are LLM, mtmd
(vision in) and tts.

The file itself says the same thing. qwen-image-2.1-F16.gguf carries no
metadata at all, so `general.architecture` is empty, and that key is what
llama.cpp dispatches on to choose a model implementation: with nothing there
it cannot even decide what to build. Its 265 tensors are named the diffusers
way (`model.diffusion_model.transformer_blocks.N.img_mlp.gate_up`), not the
llama.cpp way (`blk.N.attn_q.weight`). It was converted for
stable-diffusion.cpp, which uses GGUF too.

Three files, not one
--------------------
A Qwen-Image GGUF is the denoiser alone. stable-diffusion.cpp needs three
pieces, and refuses to generate without them:

    --diffusion-model   the GGUF Zoomies lists in the dropdown
    --vae               qwen_image_2.1_vae_bf16.safetensors
    --llm               a Qwen3-VL-8B-Instruct GGUF (the text encoder)

Qwen-Image 2.1 uses Qwen3-VL-8B, where the original Qwen-Image used
Qwen2.5-VL, and the two generations' VAEs are NOT interchangeable. Picking
the wrong pair produces noise rather than an error, which is why
`companions()` matches on the model's own version string and says what is
missing instead of guessing.

Image editing additionally wants `--llm_vision <mmproj>` and a reference
image; that is a per-request concern, so it belongs in Extra flags rather
than here.

Why sd-server and not the one-shot CLI
--------------------------------------
Zoomies is built around a long-lived server on a port: list_loaded asks the
port what it is holding, Processes stops it, and VRAM accounting follows the
process. `sd-cli` runs once and exits, which fits none of that. `sd-server`
holds the weights and serves an OpenAI-shaped API, so it drops into the same
machinery as llama.cpp:

    POST /v1/images/generations     prompt, n, size, output_format
    POST /v1/images/edits
    GET  /v1/models                 what is loaded
    GET  /                          its own web UI

Sampler, steps, cfg scale and size are request-time options, not launch
flags, so this backend deliberately exposes none of them as fields - see
supports(). The server's own page is the quickest way to drive it.
"""

import os
import re
import time

import backends
import gguf
import runner
import state

SDCPP_HOST = "127.0.0.1"
SDCPP_PORT = 1234              # sd-server's own default, and the first we try

# Zoomies owns the model paths, the listen address and the port; letting any
# of these through Extra flags would mean the dropdown and the running server
# disagree about what is loaded, or about where to reach it.
#
# sd-server spells the address flags its own way - `-l/--listen-ip` and
# `--listen-port`, not llama.cpp's `--host`/`--port`. Verified against
# `sd-server --help` on the master-945-a1ded76 Vulkan build; the llama.cpp
# names are denied too, so pasting them in gets a note rather than a server
# that quietly ignores them.
SDCPP_DENIED = (
    "--diffusion-model", "-m", "--model", "--vae", "--llm",
    "-l", "--listen-ip", "--listen-port", "--host", "--port",
)

# A diffusion GGUF is identified by its tensor names, not its metadata,
# because the conversions carry no metadata to read.
DIFFUSION_PREFIX = "model.diffusion_model."

# Files that live in the same folder but are never the thing to load: the
# denoiser's companions, and llama.cpp's own sidecars.
NOT_A_DIFFUSION_MODEL = re.compile(
    r"(vae|mmproj|text_encoder|qwen\d(\.\d)?-?vl|clip|t5xxl|-draft|ggml-vocab)",
    re.IGNORECASE)


def find_sd_exe():
    """sd-server from a stable-diffusion.cpp release, or "" if not installed.

    sd-server is looked for before sd/sd-cli: the one-shot CLI cannot hold a
    model, so finding it would let the dropdown offer a load that can never
    stay up. Mirrors find_llama_exe, including the WindowsApps directory,
    since a packaged build would land there.
    """
    dirs = (os.environ.get("PATH") or "").split(os.pathsep)
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        dirs += [
            os.path.join(local, "Microsoft", "WindowsApps"),
            os.path.join(local, "stable-diffusion.cpp"),
            os.path.join(local, "sd.cpp"),
        ]
    for name in ("sd-server.exe", "sd-server"):
        for directory in dirs:
            directory = directory.strip('"')
            if not directory:
                continue
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate
    return ""


def is_diffusion_gguf(path):
    """True when this .gguf holds a denoiser rather than a language model.

    Reads the header only - the tensor index, not the weights - so this is
    cheap enough to run over a folder and never touches the GPU.
    """
    if NOT_A_DIFFUSION_MODEL.search(os.path.basename(path)):
        return False
    try:
        header = gguf.read(path)
    except (OSError, ValueError):
        return False
    if header.arch:                  # a named architecture means llama.cpp's
        return False
    names = list(header.other_bytes) + list(header.layer_bytes)
    return any(str(n).startswith(DIFFUSION_PREFIX) for n in names)


class Companions:
    """The VAE and text encoder a denoiser needs, and what is missing.

    Kept as an object rather than a tuple because the launch notes, the
    dropdown description and is_available all want to say the same thing
    about the same model, and three copies of the matching rules drifted
    apart the first time this was written as a helper function.
    """

    def __init__(self, vae="", llm="", version="", vision=""):
        self.vae = vae
        self.llm = llm
        self.version = version
        # The projector that lets the text encoder look at a picture. Not
        # part of `complete`: without it the model still generates from a
        # prompt, and only instruction editing ("make the car red") is out
        # of reach. Plain redraw-this-picture works either way, because that
        # happens in the denoiser and never reaches the encoder.
        self.vision = vision

    @property
    def complete(self):
        return bool(self.vae and self.llm)

    def missing(self):
        out = []
        if not self.vae:
            out.append("a VAE (qwen_image_%s_vae_bf16.safetensors)"
                       % (self.version or "2.1"))
        if not self.llm:
            out.append("a text encoder (%s GGUF)" % self.encoder_name())
        return out

    def encoder_name(self):
        """Which text encoder this generation wants.

        Qwen-Image 2.1 moved to Qwen3-VL-8B; the original used Qwen2.5-VL-7B.
        Naming the right one matters because the wrong encoder loads happily
        and then conditions on nonsense.
        """
        return "Qwen3-VL-8B-Instruct" if self.version.startswith("2.1") \
            else "Qwen2.5-VL-7B-Instruct"


VL_FAMILY = re.compile(r"qwen\d(?:\.\d)?-?vl", re.IGNORECASE)


def _family(text):
    """The VL family in a filename, flattened so qwen3-vl == qwen3vl."""
    m = VL_FAMILY.search(text or "")
    return m.group(0).lower().replace("-", "") if m else ""


def companions_for(path):
    """Find the VAE, text encoder and vision projector beside a denoiser."""
    folder = os.path.dirname(path) or "."
    name = os.path.basename(path).lower()
    version = "2.1" if "2.1" in name else ("2512" if "2512" in name else "")
    try:
        entries = os.listdir(folder)
    except OSError:
        return Companions(version=version)

    vae = llm = ""
    for entry in entries:
        low = entry.lower()
        full = os.path.join(folder, entry)
        # The VAE ships as safetensors; sd.cpp reads it directly, and the
        # version has to match the denoiser's or the decode is garbage.
        if not vae and "vae" in low and low.endswith(".safetensors"):
            if not version or version in low:
                vae = full
        # The text encoder is a GGUF whose name says which VL model it is.
        if not llm and low.endswith(".gguf") and VL_FAMILY.search(low) \
                and "mmproj" not in low:
            llm = full

    # The projector has to belong to the encoder that was picked. This folder
    # also holds a gemma projector for the llama.cpp vision preset, and
    # handing that to a Qwen encoder would load a projector trained against a
    # different model - accepted at load, nonsense at inference.
    vision = ""
    want = _family(llm)
    if want:
        for entry in entries:
            low = entry.lower()
            if low.endswith(".gguf") and "mmproj" in low and _family(low) == want:
                vision = os.path.join(folder, entry)
                break
    return Companions(vae=vae, llm=llm, version=version, vision=vision)


class SdCppBackend(backends.Backend):
    name = "sdcpp"
    display_name = "stable-diffusion.cpp"
    uses_model_folder = True
    host = SDCPP_HOST
    port = SDCPP_PORT
    vram_note = ("not estimated - sd-server sizes itself from the three "
                 "files it loads, and places them with --backend")

    def __init__(self):
        backends.Backend.__init__(self)
        self._exe = find_sd_exe()
        self._exe_at = time.time() if self._exe else 0.0
        self.default_folder = backends.hf_hub_dir()
        self.endpoint = "http://%s:%d" % (SDCPP_HOST, SDCPP_PORT)

    @property
    def exe(self):
        """Where sd-server is, looked up again while it has not been found.

        Unlike llama.cpp, this is a thing people install *after* meeting the
        backend - the button is what tells them it is missing. Resolving once
        in __init__ meant the app went on saying "not found" until it was
        restarted, long after the install had happened.

        Only the empty case is re-checked, and at most every few seconds, so
        the two-second poll does not walk PATH continuously. Once found the
        answer sticks: an installed binary does not move while the app runs.
        """
        if not self._exe and (time.time() - self._exe_at) > 5.0:
            self._exe_at = time.time()
            self._exe = find_sd_exe()
        return self._exe

    @exe.setter
    def exe(self, value):
        self._exe = value
        self._exe_at = time.time()

    # -- availability ------------------------------------------------------

    def is_available(self):
        if not self.exe:
            return False, ("stable-diffusion.cpp (sd-server) not found - "
                           "install it and put sd-server.exe on PATH")
        return True, "installed"

    def can_download(self):
        """sd-server has no downloader of its own; the weights come by hand."""
        return False

    # -- discovery ---------------------------------------------------------

    def list_models(self, folder=None):
        """Diffusion GGUFs in the model folder.

        Only the denoiser is offered. The VAE and text encoder are found for
        it by companions_for() rather than listed, because neither is
        something you load on its own.
        """
        folder = folder or self.default_folder
        if not folder or not os.path.isdir(folder):
            return []
        out = []
        for entry in sorted(os.listdir(folder)):
            if not entry.lower().endswith(".gguf"):
                continue
            path = os.path.join(folder, entry)
            if not os.path.isfile(path) or not is_diffusion_gguf(path):
                continue
            label = entry[:-5]
            quant = backends.QUANT_RE.search(entry)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            out.append(backends.ModelRecord(
                backend=self.name, id=path, label=label,
                quant=quant.group(1) if quant else "",
                size_bytes=size, source="folder", gguf_path=path,
                capabilities=("image",)))
        return out

    # -- what the GUI may offer -------------------------------------------

    def supports(self, key):
        """Almost nothing: these are all language-model settings.

        Steps, cfg scale, sampler and image size are per-request options on
        /v1/images/generations, not launch flags, so there is nothing for a
        launch-time field to set. Saying so here greys each one out with the
        reason, which is better than accepting a number and dropping it.
        """
        return {
            "extra_flags": "passed to sd-server",
        }.get(key, "")

    def fixed_kv_cache(self):
        """No KV cache exists here - a denoiser keeps no conversation."""
        return None

    # -- what is loaded ----------------------------------------------------

    def _servers(self, session=None):
        session = session if session is not None else state.load_session()
        return (session.get(self.name) or {}).get("servers") or {}

    def list_loaded(self):
        """Every sd-server that answers, named by the server itself.

        Same reasoning as the llama.cpp backend: ports get reused, so the
        record written at launch goes stale and the server is the only honest
        source for what it is holding.
        """
        out = []
        records = {}
        for rec in self._servers().values():
            port = int(rec.get("port") or 0)
            if port:
                records[port] = rec
        ports = sorted(set(records) | {SDCPP_PORT})
        for port in ports:
            if not state.port_open(SDCPP_HOST, port):
                continue
            base = "http://%s:%d" % (SDCPP_HOST, port)
            name = self.model_on_port(port)
            rec = records.get(port) or {}
            if not name:
                name = rec.get("label") or rec.get("model_id") or ""
            if not name:
                continue
            out.append(backends.LoadedModel(
                backend=self.name, id=rec.get("model_id") or name, label=name,
                endpoint=base, log_path=rec.get("log") or "",
                owned_by_us=bool(rec),
                # No context: a diffusion server has no window, and leaving
                # this zero is what keeps the opencode, dsh and hermes syncs
                # from treating an image server as a chat model.
                context=0))
        return out

    def model_on_port(self, port):
        if not port:
            return ""
        listing, err = backends.http_json(
            "http://%s:%d/v1/models" % (SDCPP_HOST, int(port)), timeout=2.0)
        if err:
            return ""
        for item in (listing or {}).get("data", []) or []:
            name = str(item.get("id") or "")
            if name:
                return os.path.basename(name)
        return ""

    def remember_server(self, session, plan, shell_pid, model=None):
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

    def free_port(self, session=None):
        for port in range(SDCPP_PORT, SDCPP_PORT + 20):
            if not state.port_open(SDCPP_HOST, port):
                return port
        return SDCPP_PORT

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
        """Run sd-server with its output in the log.

        Same stderr handling as the llama.cpp backend: PowerShell 5.1 turns
        every stderr line of a native program into an error record, and
        sd.cpp logs progress there, so 'Stop' would end the script on the
        first line of output.
        """
        return "\n".join([
            "$exe = %s" % runner.ps_single(self.exe),
            "$a = @(",
            "\n".join("  " + a for a in args),
            ")",
            "Write-Log ('%s: ' + ($a -join ' '))" % what,
            "$ErrorActionPreference = 'Continue'",
            "& $exe @a 2>&1 | ForEach-Object { \"$_\" | Out-File -FilePath $log "
            "-Append -Encoding utf8 }",
            "$code = $LASTEXITCODE",
            "Write-Log ('sd-server exited with ' + $code)",
            "exit $code",
            "",
        ])

    def build_launch(self, model, settings, session, source_note=""):
        port = self.free_port(session)
        base = "http://%s:%d" % (SDCPP_HOST, port)
        script_path, log_path = runner.new_paths(self.name, model.label)
        extra, dropped = backends.split_extra_flags(settings, SDCPP_DENIED)
        path = model.gguf_path or model.id
        kit = companions_for(path)

        notes = ["Starts sd-server on port %d. Images come from "
                 "POST %s/v1/images/generations; its own page is at %s."
                 % (port, base, base)]
        if dropped:
            notes.append("Ignoring %s - Zoomies sets the model files, host "
                         "and port itself." % ", ".join(dropped))

        a = ["'--diffusion-model'", runner.ps_single(path)]
        if kit.vae:
            a += ["'--vae'", runner.ps_single(kit.vae)]
        if kit.llm:
            a += ["'--llm'", runner.ps_single(kit.llm)]
        if kit.vision:
            a += ["'--llm_vision'", runner.ps_single(kit.vision)]
        a += [runner.ps_single(t) for t in extra]
        a += ["'--listen-ip'", runner.ps_single(SDCPP_HOST),
              "'--listen-port'", runner.ps_single(port)]

        # Said once, before the branches below pick which blocker stops the
        # launch: a missing binary and missing weights are both worth knowing
        # about on the first try, rather than one per attempt.
        if not any(t.split("=", 1)[0] == "--backend" for t in extra):
            notes.extend(self._device_notes())

        if kit.complete:
            notes.append("VAE: %s. Text encoder: %s."
                         % (os.path.basename(kit.vae), os.path.basename(kit.llm)))
            if kit.vision:
                notes.append("Editing by instruction is on: %s lets the "
                             "encoder see the picture you upload. Redrawing "
                             "a picture (-i / --strength) needs no projector."
                             % os.path.basename(kit.vision))
            else:
                notes.append("Text to image and redrawing a picture only. "
                             "Editing by instruction needs a %s projector "
                             "(mmproj) beside the model." % kit.encoder_name())
        else:
            notes.append("%s is the denoiser only. It still needs %s, in the "
                         "same folder." % (model.label,
                                           " and ".join(kit.missing())))

        if not self.exe:
            body = "\n".join([
                self._preamble(log_path, "start sd-server", model.label,
                               source_note),
                "Write-Log 'stable-diffusion.cpp (sd-server) was not found'",
                "exit 2", ""])
            notes.append("stable-diffusion.cpp is not installed, so this "
                         "would only write to the log. Install sd-server and "
                         "put it on PATH.")
        elif not kit.complete:
            # Refuse rather than start: a denoiser with no VAE cannot decode a
            # latent, and with no text encoder it has nothing to condition on.
            # Launching anyway would hold 14 GB of VRAM and fail every request.
            body = "\n".join([
                self._preamble(log_path, "start sd-server", model.label,
                               source_note),
                "Write-Log %s" % runner.ps_single(
                    "cannot start: this GGUF is the denoiser only and still "
                    "needs " + " and ".join(kit.missing())),
                "exit 2", ""])
            notes.append("Nothing is started until those are there - a "
                         "denoiser with no VAE cannot decode a latent, and "
                         "with no text encoder it has nothing to work from.")
        else:
            body = "\n".join([
                self._preamble(log_path, "start sd-server", model.label,
                               source_note),
                self._native_call(a, "starting")])

        return runner.LaunchPlan(
            kind="load", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path, endpoint=base,
            host=SDCPP_HOST, port=port, long_lived=True, notes=notes,
            model_id=path, model_label=model.label,
            ready_check=lambda: self._ready(port))

    @staticmethod
    def _device_notes():
        """Say which Vulkan devices are worth using, without pinning any.

        sd-server has no device listing of its own, so the numbering is read
        from llama.cpp's - both sit on ggml-vulkan, so they enumerate the
        same adapters in the same order. It is only ever reported, never
        written into the launch: these indices are reshuffled on reboot, so a
        number baked into a preset eventually names a different adapter.

        The point of saying it at all is that an integrated GPU can sit in
        the middle of the list, which makes the obvious `vulkan0&vulkan1`
        put half the weights on a chip sharing system memory.
        """
        try:
            lc = backends.get("llamacpp")
            devices = backends.list_devices(lc.exe, lc.prefix) if lc else []
        except Exception:                             # noqa: BLE001
            return []
        if not devices:
            return []
        discrete = [(tag, desc) for tag, desc in devices
                    if not backends.INTEGRATED_GPU.search(desc)]
        shared = [(tag, desc) for tag, desc in devices
                  if backends.INTEGRATED_GPU.search(desc)]
        notes = []
        if shared:
            notes.append(
                "Leave %s out of --backend: %s shares system memory. The "
                "cards worth using are %s."
                % (", ".join(t for t, _ in shared),
                   ", ".join(d for _, d in shared),
                   ", ".join("%s (%s)" % (t, d) for t, d in discrete)))
        if len(discrete) > 1:
            notes.append(
                "To spread one model over both cards: --backend "
                "\"diffusion=%s\" --split-mode layer. Check these numbers "
                "after a reboot - Vulkan reorders them."
                % "&".join(t.lower() for t, _ in discrete))
        return notes

    def _ready(self, port):
        """Up when it can name what it loaded.

        sd-server has no /health, and the weights take a while to come in, so
        /v1/models answering at all is the signal.
        """
        listing, err = backends.http_json(
            "http://%s:%d/v1/models" % (SDCPP_HOST, port), timeout=3.0)
        return not err and bool((listing or {}).get("data"))

    def build_stop(self, loaded, session):
        port = 0
        m = re.search(r":(\d+)$", loaded.endpoint or "")
        if m:
            port = int(m.group(1))
        record = self._servers(session).get(str(port)) if port else None
        script_path, log_path = runner.new_paths(self.name,
                                                 "stop-" + loaded.label)
        shell_pid = int((record or {}).get("shell_pid") or 0)
        notes = ["Stops the sd-server on port %d." % port]
        if not record:
            notes.append("This server was not started by Zoomies.")
        body = "\n".join([
            self._preamble(log_path, "stop sd-server", loaded.label, ""),
            "$ErrorActionPreference = 'Continue'",
            "$port = %d" % port,
            "$pids = @(%d)" % shell_pid,
            "$pids += @(Get-NetTCPConnection -LocalPort $port -State Listen "
            "-ErrorAction SilentlyContinue | ForEach-Object { $_.OwningProcess })",
            "foreach ($p in ($pids | Where-Object { $_ } | Select-Object -Unique)) {",
            "  $proc = Get-Process -Id $p -ErrorAction SilentlyContinue",
            "  # A remembered pid can be reused by an unrelated program after",
            "  # a restart; only ever kill something that is plausibly ours.",
            "  if ($proc -and $proc.ProcessName -in @('powershell','sd-server','sd')) {",
            "    taskkill /T /F /PID $p 2>&1 | ForEach-Object { \"$_\" | "
            "Out-File -FilePath $log -Append -Encoding utf8 }",
            "  }",
            "}",
            "$n = 0",
            "while ((Get-NetTCPConnection -LocalPort $port -State Listen "
            "-ErrorAction SilentlyContinue) -and $n -lt 40) "
            "{ Start-Sleep -Milliseconds 250; $n++ }",
            "if (Get-NetTCPConnection -LocalPort $port -State Listen "
            "-ErrorAction SilentlyContinue) {",
            "  Write-Log 'port is still in use'; exit 1",
            "}",
            "Write-Log 'stopped'",
            "exit 0",
            "",
        ])
        return runner.LaunchPlan(
            kind="stop", backend=self.name, script_text=body,
            script_path=script_path, log_path=log_path,
            endpoint=loaded.endpoint, host=SDCPP_HOST, port=port, notes=notes)


# Registered here rather than from backends.py, so that importing either
# module first ends with the backend in REGISTRY. See the note at the bottom
# of backends.py.
backends.register(SdCppBackend())
