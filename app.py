r"""
Zoomies - LLM optimizer and agnostic launcher.

Run with:  python app.py      (or double-click Zoomies.cmd)

Threading model, copied from the Ollama Monitor because it works:
  - worker threads only ever mutate a lock-guarded dict, or push strings onto
    a Queue. No worker touches a tk widget, ever.
  - the GUI thread snapshots that dict every 200ms in refresh() and redraws.
  - a single threading.Event is what every loop waits on instead of sleeping,
    so shutdown is immediate rather than up to one poll interval late.
"""

import argparse
import ctypes
import ctypes.wintypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.parse
import webbrowser
from tkinter import filedialog, messagebox, simpledialog, ttk

import backends
import metrics
import opencode
import processes
import reasoning
import runner
import state
import ui_classic
import ui_compact
import vram
from ui_common import (BG, BG_PANEL, BORDER, FG, FONT, FONT_MONO,
                       PresetDialog, apply_style)

try:
    import optimizer
except ImportError:          # the optimizer phase has not landed yet
    optimizer = None

APP_TITLE = "Zoomies"

# Every layout draws the same controller. The name is what config.json and
# --layout use; the first entry is the default.
LAYOUTS = {
    "classic": ui_classic.ClassicLayout,
    "compact": ui_compact.CompactLayout,
}

# What a layout switch carries to the next process, deleted once read.
HANDOFF_PATH = os.path.join(state.ROOT, "handoff.json")

CANCELLED = "cancelled"          # RunResult.message of a load stopped by Cancel

POLL_SECONDS = 2.0
REFRESH_MS = 200

# How llama.cpp is told which share of the layers each card takes. Held here
# because the spill alert both writes one and has to drop the one already in
# Extra flags before it does.
SPLIT_FLAGS = {"-ts", "--tensor-split"}


def _number_or_text(value):
    """"65536" -> 65536, "0.05" -> 0.05, anything else as typed - so a saved
    preset reads like the hand-written ones."""
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def enable_dpi_awareness():
    """Must run before the first Tk window exists.

    Without this, Windows renders the whole app at the desktop scale factor by
    stretching a 96-DPI bitmap. On this machine that is 250%, which means
    every label is visibly soft and the window comes out two and a half times
    the size it asked for. Declaring awareness gets us real pixels; the
    scaling is then put back deliberately in _build_style so text stays the
    right physical size.
    """
    for call in (
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),   # per-monitor v2
        lambda: ctypes.windll.user32.SetProcessDPIAware(),        # Vista+ fallback
    ):
        try:
            call()
            return True
        except (AttributeError, OSError):
            continue
    return False


def window_dpi(widget):
    try:
        hwnd = ctypes.windll.user32.GetParent(widget.winfo_id())
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd or widget.winfo_id())
        if dpi:
            return float(dpi)
    except (AttributeError, OSError, tk.TclError):
        pass
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)   # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return float(dpi or 96)
    except (AttributeError, OSError):
        return 96.0


class Either:
    """Set when either of two Events is: shutdown, or a load's Cancel."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def is_set(self):
        return self.a.is_set() or self.b.is_set()

    def wait(self, timeout):
        end = time.time() + timeout
        while not self.is_set() and time.time() < end:
            time.sleep(0.1)
        return self.is_set()


def work_area():
    """(left, top, right, bottom) of the main screen less the taskbar, in
    real pixels, or None if Windows will not say."""
    rect = ctypes.wintypes.RECT()
    try:
        if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right, rect.bottom
    except (AttributeError, OSError):
        pass
    return None


def geometry_on_screen(geometry):
    """True if a saved WxH+X+Y still lands on a monitor - the one it was
    saved on may since have been unplugged."""
    m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$", geometry or "")
    if not m:
        return False
    w, _h, x, y = (int(n) for n in m.groups())
    try:
        # The middle of the top edge: that is where the title bar is, and
        # a window whose title bar is reachable can always be dragged back.
        point = ctypes.wintypes.POINT(x + w // 2, y + 10)
        return bool(ctypes.windll.user32.MonitorFromPoint(point, 0))
    except (AttributeError, OSError):
        return False


def dark_titlebar(root):
    """Windows 10/11 only, and entirely cosmetic - a white title bar above a
    near-black window looks broken."""
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        for attribute in (20, 19):        # 20 on current builds, 19 on older
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(value),
                    ctypes.sizeof(value)) == 0:
                return True
    except (AttributeError, OSError, tk.TclError):
        pass
    return False


class Zoomies:
    def __init__(self, root, layout=None):
        self.root = root
        self.cfg = state.load_config()
        self.session = state.load_session()
        if layout in LAYOUTS:
            self.cfg["layout"] = layout           # asked for by name: keep it
        else:
            layout = self.cfg.get("layout")
        self.layout_name = layout if layout in LAYOUTS else next(iter(LAYOUTS))
        self._restarting = False
        self._unload_now = False          # "Unload all and exit" was chosen
        self._first_poll_seen = False
        self._cancel = None               # Event for the load in progress
        self._launch_pid = 0              # its PowerShell, once running

        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.shared = {"loaded": [], "status": {}, "models": []}
        self.out_queue = queue.Queue()

        self.busy = False
        self.models = []
        # Model lists per (backend, folder), so flipping backends shows the
        # last known list instantly while a fresh one loads in the background.
        self.model_cache = {}
        self._model_req = 0
        self._announce_models = False
        self._model_index = -1            # into self.models; -1 when none
        self.loaded_items = []            # LoadedModels as last shown
        # port -> the model last seen on it, so a request can still be named
        # after the server that served it has gone.
        self._port_names = {}
        self.current_plan = None
        self.after_id = None
        self.source_note = ""
        self.doc_line = ""
        self.scale = 1.0
        self.mode_keys = {}
        self.reasoning = None             # reasoning.Spec for the chosen model
        self._reasoning_key = None        # (model, extra flags) it was read for
        self._pending_reason = None       # level to keep across a mode re-read
        self._kv_choice = backends.KV_CACHE_START     # remembered across toggles
        self._reasoning_is_variant = False
        self.opt_result = None
        self._vram_rows = {}              # luid -> DoubleVar, GB used by others
        self._vram_manual = set()         # sliders the user has moved
        self._vram_req = 0
        self._vram_after = None
        self._reasoning_after = None      # debounce for the Extra flags watch
        self._intent_tried = False        # the preset's recipe was sought
        self._vram_est = None
        self._vram_adapters = vram.adapters()
        self._spill_alerted = set()       # pids already alerted
        self._vram_limits = state.load_vram_limits()
        self._vram_watch = {}             # per card, is its footprint still
                                          # growing - see state.note_vram
        self._vram_learned = None         # the counter sample last learned
        self._spill_status = False        # the status line is ours to clear
        self._vram_seen = None            # the counter sample last followed
        self._filling = False             # see _set_value
        self._active_preset = None        # the preset under the docs values
        self._preset_fresh = False        # its docs lookup has not landed yet
        self._looking_up = False          # a docs lookup is running
        self._auto_lookup = False         # started by picking a model

        self.procs = []
        self._procs_busy = False

        self.metrics = None
        layout_cls = LAYOUTS[self.layout_name]
        self._build_style(layout_cls)
        self._build_vars()
        self.view = layout_cls(self)
        self._wire()

        state.ensure_dirs()
        state.sweep_old_files(state.SCRIPT_DIR)
        state.sweep_old_files(state.LOG_DIR)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(60, self._startup_checks)

        self.metrics = metrics.Metrics(
            self.shutdown, model_namer=self._live_model_name).start()

        self.workers = [
            threading.Thread(target=self._poll_loop, daemon=True, name="poll"),
        ]
        for w in self.workers:
            w.start()

        self.refresh()

    # ------------------------------------------------------------------
    # chrome
    # ------------------------------------------------------------------

    def px(self, n):
        """Design sizes are written for a 100% display; multiply for this one."""
        return int(round(n * self.scale))

    def _build_style(self, layout_cls):
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG)

        dpi = window_dpi(self.root)
        self.scale = max(1.0, dpi / 96.0)
        # Tk sizes fonts in points, and "tk scaling" is pixels-per-point. Set
        # it from the real DPI and every point-sized font lands at the right
        # physical size. Raw pixel measurements still need px() by hand.
        self.root.tk.call("tk", "scaling", dpi / 72.0)

        want_w, want_h = (self.px(n) for n in layout_cls.size)
        min_w, min_h = (self.px(n) for n in layout_cls.min_size)
        max_w = int(self.root.winfo_screenwidth() * 0.92)
        max_h = int(self.root.winfo_screenheight() * 0.92)
        width, height = min(want_w, max_w), min(want_h, max_h)
        saved = self.cfg.get("geometry_" + layout_cls.name) \
            if layout_cls.remember_geometry else None
        area = work_area()
        if saved and geometry_on_screen(saved):
            self.root.geometry(saved)
        elif layout_cls.dock == "right" and area:
            left, top, right, bottom = area
            height = min(want_h, bottom - top - self.px(40))   # title bar
            self.root.geometry("%dx%d+%d+%d" % (width, height,
                                                right - width - self.px(16), top))
        elif area:
            # Fit the desktop rather than the whole screen: a window taller
            # than the work area hangs below the taskbar, and its scrollbar
            # goes down there with it.
            left, top, right, bottom = area
            height = min(height, bottom - top - self.px(40))   # title bar
            self.root.geometry("%dx%d+%d+%d" % (
                width, height,
                left + max(0, (right - left - width) // 2),
                top + max(0, (bottom - top - height) // 3)))
        else:
            self.root.geometry("%dx%d+%d+%d" % (
                width, height,
                max(0, (self.root.winfo_screenwidth() - width) // 2),
                max(0, (self.root.winfo_screenheight() - height) // 3)))
        self.root.minsize(min(min_w, max_w), min(min_h, max_h))
        dark_titlebar(self.root)
        apply_style(self.root, self.px)

    def _build_vars(self):
        """Everything the form holds, owned here rather than by a layout, so
        the logic reads the same values whichever layout draws them."""
        self.on_top = tk.BooleanVar(value=self.cfg.get("always_on_top", False))
        self.unload_exit = tk.BooleanVar(value=self.cfg.get("unload_on_exit", False))
        self.one_at_a_time = tk.BooleanVar(
            value=self.cfg.get("one_model_at_a_time", True))

        last = self.cfg.get("last_backend", "ollama")
        if last not in backends.REGISTRY:       # e.g. the removed Unsloth backend
            last = "ollama"
        self.backend_var = tk.StringVar(value=last)
        self.folder_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.reason_var = tk.StringVar()
        self.preset_var = tk.StringVar()
        self.kv_var = tk.StringVar(value=backends.KV_CACHE_START)

        # What opencode was last told about the running servers, so the
        # poll only opens its config when something has actually changed.
        self._opencode_seen = None
        self._opencode_moan = ""

        self.vars, self.dirty = {}, {}
        keys = [k for row in backends.SETTING_ROWS for k in row if k]
        for key in keys + list(backends.SETTING_WIDE):
            var = tk.StringVar()
            var.trace_add("write", lambda *a, k=key: self._mark_dirty(k))
            self.vars[key] = var
            self.dirty[key] = False

    def _wire(self):
        """Hooks that need both the variables and a layout to report to."""
        for key in ("context_length", "gpu_layers", "parallel", "extra_flags"):
            self.vars[key].trace_add("write", lambda *a: self._schedule_vram())
        # --chat-template-file swaps the template the server renders, and the
        # template is what decides the reasoning levels. Watching the flags
        # means the Reasoning list follows them on its own, instead of
        # standing there stale until something else happened to refresh it.
        self.vars["extra_flags"].trace_add(
            "write", lambda *a: self._schedule_reasoning())
        self.kv_var.trace_add("write", lambda *a: self._schedule_vram())
        # With it on, the model loaded now is unloaded first, so its VRAM
        # stops counting as "in use by other apps".
        self.one_at_a_time.trace_add("write", lambda *a: self._vram_follow_live(True))

        handoff = self._read_handoff()
        if handoff.get("backend") in backends.REGISTRY:
            self.backend_var.set(handoff["backend"])
        if handoff.get("kv"):
            self._kv_choice = handoff["kv"]
        self._handoff_model = handoff.get("model_id") or ""
        # Typed values go in as typed (white, kept across a model change);
        # docs and preset values are simply looked up again.
        for key, value in (handoff.get("typed") or {}).items():
            if key in self.vars:
                self.vars[key].set(value)

        self._refresh_apply_btn()
        self._on_backend_change(initial=True)
        self._toggle_top()

    # ------------------------------------------------------------------
    # layouts
    # ------------------------------------------------------------------

    def other_layouts(self):
        return [name for name in LAYOUTS if name != self.layout_name]

    def switch_layout(self, name):
        """Reopen Zoomies in another layout.

        A restart rather than rebuilding the window in place: the worker
        threads write to the panels as they run, and swapping every widget
        under them is far riskier than a second of start-up. Loaded models
        keep running - they always outlive the window - and whatever you
        typed into the form is handed to the new process.
        """
        if name not in LAYOUTS or name == self.layout_name:
            return
        if self.busy and not messagebox.askyesno(
                "Switch layout?",
                "Something is still loading or unloading. Its output would "
                "stop showing here, though the load itself carries on.\n\n"
                "Switch anyway?", parent=self.root):
            return
        self.cfg["layout"] = name
        model = self.selected_model()
        state.write_json(HANDOFF_PATH, {
            "backend": self.backend_var.get(),
            "model_id": model.id if model else "",
            "kv": self._kv_choice,
            "typed": {k: v.get() for k, v in self.vars.items()
                      if self.dirty.get(k) and v.get().strip()},
        })
        self._restarting = True
        self._on_close()

    @staticmethod
    def _read_handoff():
        data = state.read_json(HANDOFF_PATH, {})
        try:
            os.remove(HANDOFF_PATH)
        except OSError:
            pass
        return data

    def _live_model_name(self, backend, port=0):
        """What to label a request with, asked as the request starts.

        The port it arrived on is the only thing tying a set of timings to a
        model - llama.cpp's timing lines carry no name at all - so that is
        what this answers from, and the last name seen on a port is kept so a
        request filed after its server has already been stopped is still named
        correctly. It used to take whichever model happened to be first in the
        loaded list, and fall back to the model dropdown when nothing was
        loaded any more, which filed a run of different models under one name.
        """
        with self.lock:
            loaded = list(self.shared["loaded"])
        for item in loaded:
            if port and self._port_of(item.endpoint) == port:
                self._port_names[port] = item.label
                return item.label
        be = backends.get(backend)
        if port and be is not None:
            try:
                name = be.model_on_port(port)
            except Exception:                      # noqa: BLE001
                name = ""
            if name:
                self._port_names[port] = name
                return name
        remembered = self._port_names.get(port, "")
        if remembered:
            return remembered
        # Ollama's runners listen on a port of their own that /api/ps never
        # mentions, so no port can ever match there; with one model loaded
        # there is nothing to be ambiguous about anyway. Last, because it is
        # the only answer here that is inferred rather than asked.
        mine = [i for i in loaded if i.backend == backend]
        return mine[0].label if len(mine) == 1 else ""

    @staticmethod
    def _port_of(endpoint):
        m = re.search(r":(\d+)$", endpoint or "")
        return int(m.group(1)) if m else 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def backend(self):
        return backends.get(self.backend_var.get()) or backends.get("ollama")

    def selected_model(self):
        idx = self._model_index
        if idx < 0 or idx >= len(self.models):
            return None
        return self.models[idx]

    def model_picked(self, index):
        """A layout reports a model chosen from its list."""
        if not 0 <= index < len(self.models):
            return
        self._model_index = index
        self._model_selected()

    def log(self, text, tag=None):
        self.view.log(text, tag)

    def set_status(self, text, style="Dim.TLabel"):
        self.view.set_status(text, style)

    def settings_dict(self):
        out = {k: v.get().strip() for k, v in self.vars.items()}
        spec, level = self.reasoning, self.reason_var.get()
        if spec and level in spec.levels and self.backend().supports("reasoning"):
            out["reasoning"] = level
            out["reasoning_style"] = "template"
        if self.backend().supports("kv_cache"):
            kv_type = backends.kv_choice_value(self.kv_var.get())
            if kv_type:
                out["kv_cache"] = kv_type
        return out

    def _mark_dirty(self, key):
        if self._filling:
            return                  # Zoomies is filling it in, not you
        self.dirty[key] = True
        self.view.set_field_auto(key, False)

    def _set_value(self, key, value, auto=True):
        """Auto-filled values render in accent blue so it is obvious at a
        glance which numbers came from the docs and which you typed."""
        # Flagged rather than detaching the trace: other watchers (the VRAM
        # estimate) still need to hear about the change.
        self._filling = True
        try:
            self.vars[key].set("" if value is None else str(value))
        finally:
            self._filling = False
        self.dirty[key] = not auto
        self.view.set_field_auto(key, auto)

    def _refresh_preset_box(self):
        """Offer the presets measured for this model on this backend."""
        model = self.selected_model()
        be = self.backend()
        # The same weights reach us as an Ollama tag, a .gguf path or a
        # Hugging Face name depending on the backend, so offer every handle
        # this model answers to and let the matcher normalise them.
        candidates = []
        if model:
            candidates = [model.id, model.gguf_path, model.label]
        self._presets = list(state.presets_for(
            [c for c in candidates if c], be.name)) if model else []
        # Shown by the job each was measured for, where it says so: picking
        # "Coding agent" reads better than remembering which saved name was
        # the one with the long context. Untagged presets keep their name.
        self._preset_labels = {}
        for preset in self._presets:
            label = backends.intent_label(preset.get("use_for")) or preset["name"]
            while label in self._preset_labels:      # two for the same job
                label = "%s (%s)" % (label, preset["name"])
            self._preset_labels[label] = preset
        self.view.show_presets(tuple(self._preset_labels))
        self.preset_var.set("")

    def _preset_chosen(self):
        """Fill the form from a saved preset, with the docs underneath.

        A preset holds what was measured here - context, layers, flags - and
        usually no sampling numbers; the docs hold the sampling numbers. So
        the preset goes in straight away, the docs answer is read (cached,
        so normally instant) and fills every field the preset leaves empty,
        and the preset wins wherever both have a value. Values arrive blue
        either way, so it stays obvious which numbers you typed.
        """
        label = self.preset_var.get()
        preset = getattr(self, "_preset_labels", {}).get(label)
        if not preset:
            return
        name = preset["name"]
        be = self.backend()
        self._active_preset = preset
        self._intent_tried = False
        self._refresh_apply_btn()
        applied, skipped = self._apply_preset_values(preset, force=True)
        level = str(preset.get("settings", {}).get("reasoning") or "")
        if level and be.supports("reasoning"):
            # Settled once the template is read for the preset's Extra flags,
            # which may name a different chat template file.
            self._pending_reason = level
            applied.append("Reasoning")
        elif level:
            skipped.append("Reasoning")

        self.source_note = "preset: %s" % name
        self.view.set_source(self._preset_text(preset))
        self._refresh_reasoning()
        if skipped:
            self.set_status("Applied %s. %s not supported by %s."
                            % (name, ", ".join(skipped), be.display_name),
                            "Warn.TLabel")
        else:
            self.set_status("Applied preset %s (%d settings)."
                            % (name, len(applied)))
        model = self.selected_model()
        if optimizer is not None and model is not None:
            self._preset_fresh = True
            self._start_lookup(model, keep_edits=True)

    def _intent_mode(self):
        """Point the docs recipe at the job the active preset is for.

        A model's docs often carry more than one set of sampling numbers -
        Qwen lists "precise coding tasks" beside "general tasks" - and
        choosing between them was the whole job of the old Mode dropdown.
        A preset that records what it is for answers that itself, so the
        recipe follows the preset instead of being a separate question.

        Returns True when it asked for a re-read, which the caller must let
        finish rather than carrying on with the numbers now on screen.
        """
        preset = self._active_preset
        wanted = backends.INTENT_MODES.get(str((preset or {}).get("use_for") or ""))
        # Once per preset: the re-read comes back through here, and a recipe
        # the docs do not actually have must not send it round again.
        if not wanted or len(self.mode_keys) < 2 or self._intent_tried:
            return False
        self._intent_tried = True
        keys = {key: label for label, key in self.mode_keys.items()}
        key = next((k for want in wanted for k in keys
                    if want in k or k in want), None)
        if key is None or key == self.mode_keys.get(self.mode_var.get()):
            return False
        self.mode_var.set(keys[key])
        self._mode_changed()
        return True

    def _apply_preset_values(self, preset, force=False):
        """Put a preset's fields into the form. force: over values you typed
        too - picking a preset is asking for its numbers. A docs re-read
        passes False, so it never undoes a value you typed."""
        be = self.backend()
        applied, skipped = [], []
        for key, value in (preset.get("settings") or {}).items():
            if key == "kv_cache":
                if be.supports("kv_cache"):
                    self.kv_var.set(str(value))
                    applied.append("KV cache")
                else:
                    skipped.append("KV cache")
                continue
            if key not in self.vars:
                continue                      # reasoning is handled apart
            if not force and self.dirty.get(key) and self.vars[key].get().strip():
                continue
            if be.supports(key):
                self._set_value(key, value, auto=True)
                applied.append(backends.SETTING_TEXT.get(key, key))
            else:
                skipped.append(backends.SETTING_TEXT.get(key, key))
        return applied, skipped

    @staticmethod
    def _preset_text(preset):
        return "Preset %s: %s" % (preset["name"], preset.get("note")
                                  or "measured on this machine.")

    def _save_preset(self):
        """Save the form as a preset for this model on this backend, so the
        same numbers come back next time by picking it.

        Everything filled in is saved - typed, from the docs or from a
        preset - because that is what Launch would use. Empty fields and
        ones this backend greys out are left out. Saving under an existing
        name replaces that preset's settings and keeps its note.

        The dialog also asks what the preset is for. That is stored with
        it, so the form can offer the job rather than the name, and so
        "Sync opencode..." knows which docs recipe the numbers came from.
        """
        model, be = self.selected_model(), self.backend()
        if model is None:
            self.set_status("Pick a model first.", "Warn.TLabel")
            return
        settings = {}
        for key, value in self.settings_dict().items():
            if not value or (key in self.vars and not be.supports(key)):
                continue
            settings[key] = _number_or_text(value)
        if not settings:
            self.set_status("Nothing to save - every field is empty.",
                            "Warn.TLabel")
            return
        active = self._active_preset or {}
        missing = [k for k in backends.WRITE_KEYS
                   if be.supports(k) and k not in settings]
        answer = PresetDialog(
            self.root, "Save as preset",
            "Saving %d settings for %s on %s." % (
                len(settings), model.label, be.display_name),
            tuple((k, l) for k, l, _ in backends.INTENTS),
            name=active.get("name", ""), intent=active.get("use_for", ""),
            note=("Nothing is filled in for %s. opencode is synced to send "
                  "only what a preset holds, so those fall back to the "
                  "server's own values." % ", ".join(
                      backends.SETTING_TEXT.get(k, k) for k in missing)
                  if missing else "")).result
        if not answer:
            return
        name, intent = answer
        candidates = [c for c in (model.id, model.gguf_path, model.label) if c]
        old = state.find_preset(candidates, be.name, name)
        if old and not messagebox.askyesno(
                "Save as preset",
                "Replace the settings in preset %s?\n\nIts note is kept." % name,
                parent=self.root):
            return
        preset = dict(old or {})
        preset.update(name=name, backend=be.name, settings=settings)
        if intent:
            preset["use_for"] = intent
        else:
            preset.pop("use_for", None)
        preset.setdefault("note", "Saved from the form on %s."
                          % time.strftime("%Y-%m-%d"))
        if not state.save_preset(candidates, preset):
            self.set_status("Could not write %s." % state.PRESETS_PATH,
                            "Warn.TLabel")
            return
        self._refresh_preset_box()
        self.preset_var.set(backends.intent_label(intent) or name)
        self._active_preset = preset
        self._intent_tried = False
        self._refresh_apply_btn()
        self.source_note = "preset: %s" % name
        self.view.set_source(self._preset_text(preset))
        self.set_status("Saved preset %s (%d settings)." % (name, len(settings)),
                        "Ok.TLabel")
        self.log("Preset %s saved for %s: %s" % (name, model.label, ", ".join(
            "%s=%s" % kv for kv in settings.items())), "note")

    def _clear_settings(self):
        for key in self.vars:
            self._set_value(key, "", auto=False)
            self.dirty[key] = False
        self.view.set_source("")
        self.view.set_notes("")
        self.view.show_modes(())
        self.mode_var.set("")
        self.mode_keys = {}
        self.opt_result = None
        self.source_note = ""
        self._active_preset = None
        self._refresh_apply_btn()
        self.preset_var.set("")
        self._pending_reason = None
        self._reasoning_is_variant = False
        self.reason_var.set("")
        self._refresh_reasoning()

    # ------------------------------------------------------------------
    # backend / model wiring
    # ------------------------------------------------------------------

    def _on_backend_change(self, initial=False):
        be = self.backend()
        self.cfg["last_backend"] = be.name

        if be.uses_model_folder:
            self.folder_var.set(self.cfg.get(self._folder_key(be))
                                or be.default_folder or "")
        else:
            self.folder_var.set("%s   (%s manages these)"
                                % (be.default_folder, be.display_name))
        can_download = getattr(be, "can_download", lambda: False)()
        self.view.set_folder_state(be.uses_model_folder, can_download)

        for key in self.vars:
            self.view.set_field_supported(key, bool(be.supports(key)))

        unsupported = [lab for key, lab in backends.SETTING_LABELS
                       if not be.supports(key)]
        if not be.supports("reasoning"):
            unsupported.append("Reasoning")
        msg = ""
        if unsupported:
            msg = "%s ignores: %s." % (be.display_name, ", ".join(unsupported))
        if not be.supports("reasoning"):
            msg += (" Reasoning is chosen per request by whichever app sends "
                    "the prompt.")
        fixed_kv = be.fixed_kv_cache()
        if fixed_kv and not be.supports("kv_cache"):
            msg += (" KV cache is fixed at %s by OLLAMA_KV_CACHE_TYPE for "
                    "every model." % fixed_kv)
        self.view.set_notes(msg)
        self._refresh_reason_box()
        self._refresh_kv_box()
        self._schedule_vram()

        self._reload_models(announce=False)

    def _reload_models(self, announce=True):
        """Refresh the model dropdown without freezing the window.

        This used to call list_models() on the UI thread, so every backend
        toggle stalled while a backend answered its requests. Now the last
        known list for that backend appears immediately and the real one is
        fetched on a worker, landing through the queue like everything else.

        Rescan says how many it found. A scan that turns up the same list as
        before is indistinguishable from a button that does nothing.
        """
        be = self.backend()
        folder = self.folder_var.get() if be.uses_model_folder else None
        key = (be.name, folder or "")
        cached = self.model_cache.get(key)
        if cached is not None:
            self._show_models(be, cached)
        else:
            self.models = []
            self._model_index = -1
            self.view.show_models(())
            self.model_var.set("Loading models...")
        self._model_req += 1
        self._announce_models = bool(announce)
        if announce:
            self.set_status("Scanning for %s models..." % be.display_name)
        threading.Thread(target=self._load_models_worker,
                         args=(be, folder, key, self._model_req),
                         daemon=True).start()

    def _load_models_worker(self, be, folder, key, req):
        try:
            models = be.list_models(folder)
        except Exception as exc:                      # noqa: BLE001
            be.last_error = str(exc)
            models = []
        self.out_queue.put(("models", key, req, models, be.last_error))

    def _models_ready(self, key, req, models, err):
        self.model_cache[key] = models
        be = self.backend()
        folder = self.folder_var.get() if be.uses_model_folder else None
        # The user may have toggled again while this was loading.
        if key != (be.name, folder or "") or req != self._model_req:
            return
        self._show_models(be, models, err)
        if self._announce_models:
            self._announce_models = False
            self.set_status(
                "%s: %d model%s." % (be.display_name, len(models),
                                     "" if len(models) == 1 else "s"),
                "Ok.TLabel" if models else "Warn.TLabel")

    def _show_models(self, be, models, err=""):
        current = self.selected_model()
        want = current.id if current else (self._handoff_model or
                                           self.cfg.get("last_model", ""))
        self.models = list(models)
        labels = [m.describe() for m in self.models]
        self.view.show_models(labels)
        chosen = next((i for i, m in enumerate(self.models) if m.id == want), 0)
        if labels:
            self._handoff_model = ""      # it has found its model; done
            self._model_index = chosen
            self.model_var.set(labels[chosen])
        else:
            self._model_index = -1
            self.model_var.set("")
            if err:
                self.log("%s: %s" % (be.display_name, err), "err")
        # A rescan that lands on the same model is not a new pick: keep the
        # form, and any preset in it.
        now = self.selected_model()
        same = current is not None and now is not None and \
            (current.backend, current.id) == (now.backend, now.id)
        if not same:
            self._model_selected()

    def _browse(self):
        chosen = filedialog.askdirectory(
            title="Pick the folder holding your model files",
            initialdir=self.folder_var.get() or os.path.expanduser("~"))
        if chosen:
            self.folder_var.set(chosen)
            self.cfg[self._folder_key(self.backend())] = chosen
            self._reload_models()

    @staticmethod
    def _folder_key(be):
        return "%s_folder" % be.name

    def _download(self):
        """Fetch a model for llama.cpp by its Hugging Face name."""
        be = self.backend()
        if self.busy or not getattr(be, "can_download", lambda: False)():
            return
        name = simpledialog.askstring(
            "Download a model",
            "Hugging Face name and quant, as llama.cpp takes it:\n\n"
            "    ggml-org/Qwen3.5-0.8B-GGUF:Q8_0\n\n"
            "Match the quant you tested on Ollama (for example Q4_K_M).",
            parent=self.root)
        name = (name or "").strip()
        if not name:
            return
        if not backends.HF_REPO_QUANT.match(name):
            self.set_status("That is not an org/repo:QUANT name.", "Warn.TLabel")
            return
        plan = be.build_download(name)
        self.current_plan = plan
        self.busy = True
        self.view.set_launch_enabled(False)
        self.set_status("Downloading %s..." % name)
        self.log("")
        self.log("=== %s ===" % os.path.basename(plan.script_path), "note")
        threading.Thread(target=self._run_worker, args=(plan, False),
                         daemon=True).start()

    def _open_folder(self):
        be = self.backend()
        path = be.default_folder if not be.uses_model_folder else self.folder_var.get()
        self._reveal(path)

    def _open_scripts(self):
        self._reveal(state.SCRIPT_DIR)

    def _open_log(self):
        plan = self.current_plan
        if plan and os.path.isfile(plan.log_path):
            self._reveal(plan.log_path)
        else:
            self._reveal(state.LOG_DIR)

    def _reveal(self, path):
        try:
            os.startfile(path)         # noqa: S606 - Windows shell open
        except OSError as exc:
            self.log("Could not open %s (%s)" % (path, exc), "err")

    def _toggle_top(self):
        self.root.attributes("-topmost", bool(self.on_top.get()))
        self.cfg["always_on_top"] = bool(self.on_top.get())

    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------

    def _apply_optimal(self, force=False):
        if optimizer is None:
            return
        model = self.selected_model()
        if model is None:
            self.set_status("Pick a model first.", "Warn.TLabel")
            return
        dirty = [k for k, d in self.dirty.items() if d and self.vars[k].get().strip()]
        if dirty and not messagebox.askyesno(
                "Overwrite your edits?",
                "You have typed values into: %s.\n\nReplace them with the "
                "recommended settings?" % ", ".join(dirty), parent=self.root):
            return
        self.set_status("Re-reading the docs..." if force
                        else "Looking up settings...")
        self._start_lookup(model, force=force)

    def _start_lookup(self, model, force=False, keep_edits=False):
        self._looking_up = True
        self._refresh_apply_btn()
        threading.Thread(target=self._apply_worker,
                         args=(model, force, keep_edits), daemon=True).start()

    def _refresh_apply_btn(self):
        """Greyed out only while a lookup is already running.

        It used to be greyed for as long as a preset was active, on the
        grounds that the preset had already filled everything in - but that
        turned picking a preset into a dead end you had to press Clear to
        escape. The docs fill in around a preset anyway: they only touch
        fields the preset left empty, and the preset goes back on top.
        """
        self.view.set_apply_enabled(optimizer is not None
                                    and not self._looking_up)

    def _apply_worker(self, model, force=False, keep_edits=False):
        """keep_edits: set when a preset asked for the docs, which must fill
        in around what you typed rather than over it. Apply itself already
        asked before replacing typed values, so it may."""
        preset = self._active_preset
        level = str(((preset or {}).get("settings") or {}).get("reasoning") or "")
        try:
            result = optimizer.recommend(model, self.cfg,
                                         force_refresh=force)
            # A preset that switches thinking off wants the docs' Instruct
            # numbers, not the Thinking ones the model defaults to.
            if level in ("off", "none") and "instruct" in result.modes                     and result.mode != "instruct":
                result = optimizer.recommend(model, self.cfg, mode="instruct")
        except Exception as exc:                      # noqa: BLE001
            result = optimizer.Result(error=str(exc))
        self.out_queue.put(("optimal", model, result, keep_edits))

    def _apply_result(self, model, result, keep_edits=False):
        """keep_edits: a Mode or Reasoning change re-reads the docs for new
        sampling numbers, but must not touch what you typed - it used to put
        the docs' context back over a context you had just raised."""
        self._looking_up = False
        self._refresh_apply_btn()
        auto, self._auto_lookup = self._auto_lookup, False
        current = self.selected_model()
        if current is None or current.id != model.id:
            return                        # a different model is picked now
        be = self.backend()
        self.opt_result = result
        preset = self._active_preset

        if not result.settings:
            if preset:
                # The preset stands on its own; no page picker for a
                # question nobody asked.
                self.view.set_source("%s\nNo docs settings found "
                                     "to fill in the rest." %
                                     self._preset_text(preset))
                return
            self.view.set_source(result.error or
                                 "Nothing usable found on that page.")
            self._reasoning_is_variant = False
            self._refresh_reason_box()
            if auto:
                # Nobody asked yet, so no dialog: say so, and leave the
                # page picker to an actual press of the button.
                self.set_status("No docs settings for this model - type them "
                                "in, or press Fill from docs to pick a page.",
                                "Warn.TLabel")
                return
            self.set_status("No settings found - type them in, or pick a page.",
                            "Warn.TLabel")
            if result.candidates:
                self._offer_page_picker(model, result.candidates)
            return

        applied, skipped = [], []
        for key, value in result.settings.items():
            if key not in self.vars:
                continue
            if keep_edits and self.dirty.get(key) and self.vars[key].get().strip():
                continue
            if be.supports(key):
                self._set_value(key, value, auto=True)
                applied.append(key)
            else:
                skipped.append(backends.SETTING_TEXT.get(key, key))

        # The dropdown shows the readable label; the lookup maps it back to
        # the internal key so flipping modes re-reads the right column.
        self.mode_keys = {result.mode_labels.get(k, k): k for k in result.modes}
        self.view.show_modes(tuple(self.mode_keys))
        if result.mode:
            self.mode_var.set(result.mode_labels.get(result.mode, result.mode))
        if self._intent_mode():
            return          # re-reading for the recipe that job wants

        # The preset goes back on top: it was measured on this machine, the
        # docs were not. Its reasoning level too, when it has one.
        from_preset = []
        if preset:
            from_preset, _ = self._apply_preset_values(preset)
            level = str((preset.get("settings") or {}).get("reasoning") or "")
            # Its level too - but not after you changed Mode or Reasoning
            # yourself, which is what a keep_edits re-read otherwise means.
            if level and (self._preset_fresh or not keep_edits):
                self._pending_reason = level
            self._preset_fresh = False

        # The levels come from the chat template, not the docs; the docs only
        # say whether a model with no switch has a separate Reasoning build.
        self._reasoning_is_variant = any(
            "reason" in str(label).lower()
            for label in result.mode_labels.values())
        self._refresh_reasoning()

        if preset:
            texts = [backends.SETTING_TEXT.get(k, k) for k in applied
                     if backends.SETTING_TEXT.get(k, k) not in from_preset]
            self.source_note = "preset: %s; %s" % (preset["name"],
                                                   result.source_line())
            self.view.set_source("%s\nFrom %s: %s. The preset's "
                                 "own values win where both have one." % (
                                     self._preset_text(preset),
                                     result.source_line(),
                                     ", ".join(texts) or "nothing new"))
            msg = "Applied preset %s (%d settings) and %d from the docs." % (
                preset["name"], len(from_preset), len(texts))
        else:
            self.source_note = result.source_line()
            self.view.set_source(result.describe())
            msg = "Applied %d setting%s." % (len(applied),
                                             "" if len(applied) == 1 else "s")
        if skipped:
            msg += "  %s cannot use: %s." % (be.display_name, ", ".join(skipped))
        self.set_status(msg, "Ok.TLabel")
        for hint in result.suggestions:
            self.log("Not applied: " + hint, "note")

    def _model_selected(self):
        """A new model fills the form from its docs straight away, so the
        previous model's numbers never sit under the new one's name. The
        answer is saved per family (cache\resolved.json), so only the first
        pick of a new family goes online. Values you typed are kept."""
        self._active_preset = None
        self._refresh_apply_btn()
        # What the last model's docs or preset filled in (blue) goes; what
        # you typed (white) stays.
        for key, var in self.vars.items():
            if not (self.dirty.get(key) and var.get().strip()):
                self._set_value(key, "", auto=True)
        self.view.set_source("")
        self.source_note = ""
        self.reason_var.set("")
        self._pending_reason = None
        self._reasoning_is_variant = False
        self._refresh_preset_box()
        self._refresh_reasoning(force=True)
        self._schedule_vram()
        model = self.selected_model()
        if optimizer is not None and model is not None:
            self._auto_lookup = True
            self._start_lookup(model, keep_edits=True)

    def _refresh_reasoning(self, force=False):
        """Read the chosen model's chat template for its reasoning levels.

        Read for the Extra flags in the form, because --chat-template-file
        swaps the template the server renders. Off the UI thread: the first
        read of a .gguf steps over its whole vocabulary.
        """
        model = self.selected_model()
        be = self.backend()
        if model is None or not be.supports("reasoning"):
            self.reasoning, self._reasoning_key = None, None
            self._refresh_reason_box()
            return
        extra = self.vars["extra_flags"].get().strip()
        key = (be.name, model.id, extra)
        if key == self._reasoning_key and not force:
            self._choose_reason()
            self._refresh_reason_box()
            return
        self._reasoning_key = key
        self.reasoning = None
        self._refresh_reason_box()
        threading.Thread(
            target=lambda: self.out_queue.put(
                ("reasoning", key, reasoning.spec_for(model, extra))),
            daemon=True).start()

    def _reasoning_ready(self, key, spec):
        if key != self._reasoning_key:
            return                    # the model or its flags changed since
        self.reasoning = spec
        self.log("Reasoning for %s: %s" % (key[1], spec.describe()),
                 "note" if spec.problem else None)
        for note in spec.notes:
            self.log("  " + note, "note")
        self._choose_reason()
        self._refresh_reason_box()

    def _choose_reason(self):
        """Reasoning follows Mode, so the two can never contradict each other:
        Instruct (non-thinking) sampling with reasoning switched off, and
        thinking sampling with the template's default level - unless a level
        was just picked or came from a preset."""
        spec = self.reasoning
        if not spec or not spec.usable:
            return
        pending, self._pending_reason = self._pending_reason, None
        mode = self.mode_keys.get(self.mode_var.get())
        current = self.reason_var.get()
        if pending in spec.levels:
            choice = pending
        elif pending in ("none", "off") and spec.off:
            choice = spec.off         # a preset saved before levels were read
        elif mode == "instruct" and spec.off:
            choice = spec.off
        elif current in spec.levels and not (mode == "thinking"
                                             and current == spec.off):
            choice = current
        else:
            choice = spec.default
        self.reason_var.set(choice)

    def _refresh_reason_box(self):
        """Offer exactly the levels this model's chat template accepts.

        Nothing is hardcoded because the scales differ: Qwen3.8 has an off
        switch and three effort levels, Muse Glimmer has four strengths and
        no off, Gemma 4 is on or off, and Ministral 3 cannot be switched at
        all - its Reasoning version is a separate download.
        """
        be = self.backend()
        spec = self.reasoning
        if not spec or not spec.usable:
            self.view.show_reasoning((), False)
            if not be.supports("reasoning") or self.selected_model() is None:
                text = ""
            elif spec is None:
                text = "reading..."
            elif spec.problem:
                text = "unknown"
            elif self._reasoning_is_variant:
                text = "separate model"
            else:
                text = "always on" if spec.thinks else "none"
            self.reason_var.set(text)
            return
        self.view.show_reasoning(spec.levels, True)
        if self.reason_var.get() not in spec.levels:
            self.reason_var.set(spec.default)

    # ------------------------------------------------------------------
    # VRAM estimate
    # ------------------------------------------------------------------

    # What the load will need per card, before loading it. Worked out from
    # the .gguf header (vram.py), so it never touches a GPU. Each card also
    # has a value for what everything else already holds on it; those follow
    # Windows' own counters until you move one.

    def _schedule_reasoning(self):
        """Re-read the template shortly after the last keystroke in Extra
        flags, not on every one: each read opens the .gguf."""
        if self._reasoning_after:
            self.root.after_cancel(self._reasoning_after)
        self._reasoning_after = self.root.after(400, self._reasoning_due)

    def _reasoning_due(self):
        self._reasoning_after = None
        self._refresh_reasoning()

    def _schedule_vram(self):
        """Re-estimate shortly after the last change, not on every key."""
        if self._vram_after:
            self.root.after_cancel(self._vram_after)
        self._vram_after = self.root.after(250, self._estimate_vram)

    def _estimate_vram(self):
        self._vram_after = None
        model, be = self.selected_model(), self.backend()
        if model is None or be.name != "llamacpp":
            self._vram_req += 1
            self._sync_vram_rows([])
            self.view.show_vram_estimate(
                None, "" if model is None else
                "estimated for llama.cpp only - Ollama places layers itself",
                "Dim.TLabel", "")
            return
        settings = self.settings_dict()
        cards = vram.cards_for(reasoning.split_flags(settings.get("extra_flags")),
                               self._vram_adapters)
        self._sync_vram_rows(cards)
        other = {luid: int(var.get() * vram.GB)
                 for luid, var in self._vram_rows.items()}
        self._vram_req += 1
        req, found = self._vram_req, self._vram_adapters
        threading.Thread(
            target=lambda: self.out_queue.put(
                ("vram", req, vram.estimate(model, settings, other, found))),
            daemon=True).start()

    def _sync_vram_rows(self, cards):
        """One value per card the launch will use."""
        if [c.luid for c in cards] == list(self._vram_rows):
            return
        live = self._live_other()
        self._vram_rows = {
            card.luid: tk.DoubleVar(value=round(live.get(card.luid, 0) / vram.GB, 1))
            for card in cards}
        self.view.show_vram_cards(cards, self._vram_rows, self._vram_slid)
        for card in cards:
            self._vram_label(card.luid)

    def _vram_label(self, luid):
        self.view.set_vram_card_text(luid, "%.1f GB%s" % (
            self._vram_rows[luid].get(), "  (set by you)"
            if luid in self._vram_manual else "  (measured)"))

    def _vram_slid(self, luid):
        self._vram_manual.add(luid)
        self._vram_label(luid)
        self._schedule_vram()

    def _live_other(self):
        """{luid: bytes} each card holds now, less any model server that
        "One model at a time" would unload before this load."""
        snap = self.metrics.snapshot() if self.metrics else {}
        used = {g["luid"]: g["used"] for g in snap.get("gpus") or []
                if g.get("used") is not None}
        if used and self.one_at_a_time.get():
            for pid, mem in (snap.get("gpu_procs") or {}).items():
                if mem["dedicated"] < 256 * 1024 ** 2 or \
                        not metrics.model_server(pid):
                    continue
                for luid, n in mem.get("by_card", {}).items():
                    if luid in used:
                        used[luid] = max(0, used[luid] - n)
        return used

    def _vram_follow_live(self, reset=False):
        """Keep untouched sliders on the measured value. reset: take the
        measured value for every slider, including ones you moved."""
        if reset:
            self._vram_manual.clear()
        elif self.metrics is not None:
            # Counters arrive every few seconds; the panel refreshes faster.
            sample = self.metrics.snapshot().get("gpu_procs")
            if sample is self._vram_seen:
                return
            self._vram_seen = sample
        live = self._live_other()
        changed = False
        for luid, var in self._vram_rows.items():
            if luid in self._vram_manual or luid not in live:
                continue
            value = round(live[luid] / vram.GB, 1)
            if abs(value - var.get()) >= 0.1 or reset:
                var.set(value)
                self._vram_label(luid)
                changed = True
        if changed:
            self._schedule_vram()

    def _vram_ready(self, req, est):
        if req != self._vram_req:
            return                        # the form changed since
        self._vram_est = est
        if est.problem:
            self.view.show_vram_estimate(None, est.problem, "Dim.TLabel", "")
            return
        # Against what each card will hand out, not its sticker VRAM.
        parts = ["%s %s + %s = %s / %s GB" % (
            c.name, vram.gb(c.other), vram.gb(c.model), vram.gb(c.used),
            vram.gb(c.limit)) for c in est.cards]
        worst = min(est.cards, key=lambda c: c.spare)
        # Tight first: it is a kind of not-fitting now, and the two read
        # very differently to someone deciding whether to press Load.
        if est.tight:
            verdict = "tight - only %s GB spare on %s" % (
                vram.gb(worst.spare), worst.name)
            style = "Warn.TLabel"
        elif not est.fits:
            verdict = "will spill about %s GB into system RAM on %s" % (
                vram.gb(-worst.spare), worst.name)
            style = "Bad.TLabel"
        else:
            verdict = "fits - %s GB spare" % vram.gb(worst.spare)
            style = "Ok.TLabel"
        notes = ["Other apps + this model = total. Accurate to about 0.25 GB "
                 "per card; keeps %s GB free per card for \"largest context\"."
                 % vram.gb(est.margin)]
        measured = [c for c in est.cards if c.budget and c.budget < c.total]
        if measured:
            notes.append(
                "Card limits are measured rather than the number on the box "
                "(%s): Windows holds the rest back for the desktop, and a "
                "model that spilled there is what showed where the line is."
                % ", ".join("%s %s of %.0f GB" % (
                    c.name, vram.gb(c.budget), c.total / float(vram.GB))
                    for c in measured))
        notes += est.notes
        self.view.show_vram_estimate(
            est, "%s   ->   %s" % ("   |   ".join(parts), verdict), style,
            "  ".join(notes), verdict=verdict)

    def _use_max_ctx(self):
        est = self._vram_est
        if est and est.max_ctx:
            self.vars["context_length"].set(str(est.max_ctx))

    def _check_spills(self, snap):
        """Alert once per server when part of a model lands in system RAM.

        Windows does not fail a load that does not fit; it backs the rest
        with shared memory and the model just runs slowly, so without this
        nothing on screen would say why.

        Which card ran out is the useful half, and the two cases want
        opposite answers: a card that is full while another still has room
        is a split to rebalance, not a model to shrink. A card is only
        named once the counters say it holds the spilled memory - the
        process totals alone cannot tell the two apart.
        """
        live_pids = set(snap.get("gpu_procs") or {})
        if live_pids:                     # not after a failed sample
            self._spill_alerted &= live_pids
        spilling = snap.get("spills") or []
        if not spilling and self._spill_status:
            # Said once, and taken back once: a status line still claiming a
            # spill after the model was reloaded smaller is worse than saying
            # nothing at all.
            self.set_status("No longer spilling into system RAM.", "Ok.TLabel")
            self._spill_status = False
        for spill in spilling:
            if spill["pid"] in self._spill_alerted:
                continue
            self._spill_alerted.add(spill["pid"])
            name = next((m.label for m in self.loaded_items
                         if m.pid == spill["pid"]), "") or os.path.basename(
                             spill["image"])
            text, split = self._spill_text(name, spill, snap)
            self.log("[zoomies] " + text, "err")
            self.set_status("%s is spilling into system RAM." % name,
                            "Bad.TLabel")
            self._spill_status = True
            self._alert("Model spilling into system RAM", text, split)

    def _learn_vram(self, snap):
        """Remember what each card actually hands out.

        A card with part of a model in system RAM is at its limit - if more
        could have been placed there it would have been - and a card
        holding one without spilling proves at least that much is fine.
        Cards with no model on them are ignored: the desktop sitting at
        3 GB says nothing about what the next 11 GB request will be given.

        This is the only way to know. Windows publishes the sticker VRAM,
        never the budget, and the gap between them is what makes an
        estimate say "fits" and a load spill anyway. What counts as a
        measurement, and why most polls during a spill are not one, is
        state.note_vram's problem.
        """
        rows = snap.get("gpus") or []
        # The counters are read every few seconds and this runs five times
        # a second, so without this the same reading would be folded in ten
        # times over - which is how "seen" came to say 1064.
        if rows is self._vram_learned:
            return
        self._vram_learned = rows
        changed = False
        for row in rows:
            if row.get("used") is None or not row.get("model_bytes"):
                continue
            if state.note_vram(self._vram_limits, row.get("luid"),
                               row.get("vram"), row["used"],
                               row.get("spilled") or 0, self._vram_watch):
                changed = True
        if changed:
            state.save_vram_limits(self._vram_limits)
            self._schedule_vram()     # whatever is on screen was worked out
                                      # against the old limits

    def _spill_text(self, name, spill, snap):
        """(what spilled and the fix it calls for, the -ts to offer).

        The flag is handed back rather than only named, because a split is
        the one fix nobody can work out in their head: it is per card, it
        is proportional, and the numbers move with whatever else is on the
        cards at the time.
        """
        cards = {g["luid"]: g for g in (snap.get("gpus") or []) if g.get("luid")}
        worst = (spill.get("cards") or [None])[0]
        where, roomy, split = "", [], ""
        if worst and worst["luid"] in cards:
            card = cards[worst["luid"]]
            where = " on %s" % card["name"]
            if card.get("used") is not None:
                where += " (at %s / %.0f GB)" % (
                    vram.gb(card["used"]), card["vram"] / float(vram.GB))
            roomy = [c for luid, c in cards.items()
                     if luid != worst["luid"] and c.get("used") is not None
                     and c["vram"] - c["used"] >= worst["shared"]]
        if roomy:
            split = self._spill_split(spill, snap)
            fix = ("%s still has %s GB free, so moving some layers there "
                   "should fix it without shrinking the model: "
                   % (roomy[0]["name"],
                      vram.gb(roomy[0]["vram"] - roomy[0]["used"])))
            fix += ("put %s in Extra flags and load again." % split if split
                    else "-ts in Extra flags decides the share each card takes.")
        else:
            fix = ("No other card has room for it, so lower Context, pick a "
                   "smaller KV cache type, or close whatever else is using "
                   "the cards.")
        return ("%s is spilling%s: %s GB of its memory is in system RAM "
                "(%s GB on the cards), so prompts and generation will be much "
                "slower. %s" % (name, where, vram.gb(spill["shared"]),
                                vram.gb(spill["dedicated"]), fix)), split

    def _spill_split(self, spill, snap):
        """A -ts giving each card a share of this model in proportion to the
        room it has for it, or "" when there is nothing sensible to say.

        This model's own bytes come off each card first: the question is
        where its layers could go, not where they sit now. The cards come
        from vram.cards_for, so the entries land in llama.cpp's own device
        order - the integrated GPU is skipped, which means there are as
        many numbers as there are rows in the VRAM panel.
        """
        cards = vram.cards_for(
            reasoning.split_flags(self.vars["extra_flags"].get()),
            self._vram_adapters)
        if len(cards) < 2:
            return ""                     # one card cannot be split over
        rows = {g["luid"]: g for g in (snap.get("gpus") or []) if g.get("luid")}
        mine = ((snap.get("gpu_procs") or {}).get(spill["pid"])
                or {}).get("by_card") or {}
        free = []
        for card in cards:
            row = rows.get(card.luid)
            if row is None or row.get("used") is None:
                return ""                 # a card the counters did not report
            other = max(0, row["used"] - int(mine.get(card.luid, 0)))
            free.append(max(0, card.limit - other))
        total = float(sum(free))
        if total <= 0:
            return ""
        # llama.cpp normalises -ts itself, so these need not sum to 1.
        return "-ts " + ",".join("%.2f" % (f / total) for f in free)

    def _apply_split(self, flag, win=None):
        """Put the recommended -ts into Extra flags, replacing one already
        there. The field's trace re-estimates on its own, so the VRAM rows
        say whether it worked before anything is loaded again."""
        kept, _ = backends.split_extra_flags(
            {"extra_flags": self.vars["extra_flags"].get()}, SPLIT_FLAGS)
        self.vars["extra_flags"].set(" ".join(kept + flag.split()))
        if win is not None:
            win.destroy()
        self.set_status("Extra flags now say %s - load again to use it."
                        % flag, "Ok.TLabel")

    def _alert(self, title, text, apply_split=""):
        """A small window that does not block the rest of the app.

        apply_split: a -ts the window offers to write into Extra flags, so
        the fix is a button rather than a number to copy out by hand.
        """
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.transient(self.root)
        ttk.Label(win, text=text, style="Bad.TLabel", wraplength=self.px(460),
                  justify="left").pack(padx=14, pady=(14, 8))
        row = ttk.Frame(win)
        row.pack(pady=(0, 12))
        if apply_split:
            ttk.Button(row, text="Apply to Extra flags",
                       command=lambda: self._apply_split(apply_split, win)
                       ).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="OK", command=win.destroy).pack(side="left")
        win.lift()
        self.root.bell()

    def _refresh_kv_box(self):
        """Selectable where the backend applies it per load (llama.cpp); on
        Ollama it shows the server-wide value, greyed out, because Ollama
        takes it from an environment variable, not from a load request."""
        be = self.backend()
        if be.supports("kv_cache"):
            self.view.show_kv(backends.KV_CACHE_CHOICES, True)
            self.kv_var.set(self._kv_choice)
        else:
            fixed = be.fixed_kv_cache()
            self.view.show_kv((), False)
            self.kv_var.set("%s (fixed)" % fixed if fixed else "")

    def _kv_changed(self):
        if backends.kv_choice_value(self.kv_var.get()) or \
                self.kv_var.get() == backends.KV_CACHE_DEFAULT:
            self._kv_choice = self.kv_var.get()

    def _reason_changed(self):
        """Switching reasoning off moves Mode to Instruct, and switching it on
        moves Mode to Thinking, so the sampling numbers always match."""
        spec, level = self.reasoning, self.reason_var.get()
        if not spec or level not in spec.levels:
            return
        want = "instruct" if level == spec.off else "thinking"
        current = self.mode_keys.get(self.mode_var.get())
        if want == current or want not in self.mode_keys.values():
            return
        self._pending_reason = level
        label = next(l for l, k in self.mode_keys.items() if k == want)
        self.mode_var.set(label)
        self._mode_changed()

    def _mode_changed(self):
        """Re-read the docs for the mode the user just picked."""
        if optimizer is None:
            return
        model = self.selected_model()
        label = self.mode_var.get()
        key = self.mode_keys.get(label)
        if model is None or not key:
            return
        self.set_status("Reading %s settings..." % label)
        threading.Thread(
            target=lambda: self.out_queue.put(
                ("optimal", model,
                 optimizer.recommend(model, self.cfg, mode=key), True)),
            daemon=True).start()

    def _offer_page_picker(self, model, candidates):
        """No source had settings for this model: no Unsloth docs page, no
        recommendation on its Hugging Face card, nothing packaged with it.

        Rather than guessing a near-match - which is how a model quietly ends
        up running another model's sampling settings - ask, or hand the
        search to the user's own browser.
        """
        win = tk.Toplevel(self.root)
        win.title("Which docs page?")
        win.configure(bg=BG)
        win.geometry("%dx%d" % (self.px(620), self.px(430)))
        win.transient(self.root)
        ttk.Label(win, wraplength=self.px(580), justify="left",
                  style="Warn.TLabel",
                  text=('No recommended settings found for "%s" - not in the '
                        "Unsloth docs, not on its Hugging Face model card, and "
                        "none packaged with the model - so nothing was filled "
                        "in.\n\nIf an Unsloth page applies, pick it here and "
                        "Zoomies will remember the choice for this model. Or "
                        "search the web and type the numbers in yourself. Leave "
                        "it alone if you are not sure - a wrong page is worse "
                        "than an empty box." % optimizer.search_name(model.id))
                  ).pack(fill="x", padx=12, pady=(12, 8))

        box = tk.Listbox(win, bg=BG_PANEL, fg=FG, font=FONT, relief="flat",
                         highlightthickness=1, highlightbackground=BORDER,
                         selectbackground="#094771")
        for title in candidates:
            box.insert("end", title)
        box.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=12, pady=(0, 12))

        def use_it():
            sel = box.curselection()
            if not sel:
                return
            optimizer.remember_page(self.cfg, model, box.get(sel[0]))
            state.save_config(self.cfg)
            win.destroy()
            self._apply_optimal()

        ttk.Button(bar, text="Use this page", style="Go.TButton",
                   command=use_it).pack(side="left")
        def search_web():
            size = " ".join(sorted(optimizer.size_tokens(model.id)))
            query = " ".join(filter(None, (
                optimizer.search_name(model.id), size,
                "recommended settings temperature top_p top_k")))
            webbrowser.open("https://duckduckgo.com/?q=" + urllib.parse.quote(query))

        ttk.Button(bar, text="Search the web",
                   command=search_web).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Cancel", command=win.destroy).pack(side="left",
                                                                 padx=(8, 0))

    # ------------------------------------------------------------------
    # load / unload
    # ------------------------------------------------------------------

    def _build_plan(self):
        model = self.selected_model()
        if model is None:
            self.set_status("Pick a model first.", "Warn.TLabel")
            return None
        be = self.backend()
        try:
            return be.build_launch(model, self.settings_dict(), self.session,
                                   source_note=self.source_note)
        except Exception as exc:                      # noqa: BLE001
            self.log("Could not build the script: %s" % exc, "err")
            return None

    def _preview(self):
        plan = self._build_plan()
        if plan is None:
            return
        self._show_text("Script preview", plan.script_text, plan.notes)

    def _show_text(self, title, body, notes=(), action=None):
        """action: (button text, callback) offered beside Close."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.geometry("%dx%d" % (self.px(900), self.px(620)))
        win.transient(self.root)
        if notes:
            ttk.Label(win, text="\n".join("- " + n for n in notes),
                      style="Warn.TLabel", wraplength=self.px(860),
                      justify="left").pack(fill="x", padx=10, pady=(10, 4))
        wrap = ttk.Frame(win)
        wrap.pack(fill="both", expand=True, padx=10, pady=10)
        txt = tk.Text(wrap, bg=BG_PANEL, fg=FG, font=FONT_MONO, relief="flat",
                      wrap="none", highlightthickness=1, highlightbackground=BORDER)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=txt.yview)
        sbx = ttk.Scrollbar(win, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=sb.set, xscrollcommand=sbx.set)
        txt.insert("1.0", body)
        txt.configure(state="disabled")
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        sbx.pack(fill="x", padx=10, pady=(0, 8))
        buttons = ttk.Frame(win)
        buttons.pack(pady=(0, 10))
        if action:
            text, callback = action
            ttk.Button(buttons, text=text, style="Go.TButton",
                       command=lambda: (win.destroy(), callback())).pack(
                side="left", padx=(0, 8))
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="left")

    def _sync_opencode(self):
        """Bring opencode's reasoning dropdowns in line with each model's
        chat template. Shows every change and asks before writing."""
        self.set_status("Reading opencode's config and every model's template...")

        def work():
            try:
                self.out_queue.put(("opencode", opencode.plan_sync(), None))
            except Exception as exc:                  # noqa: BLE001
                self.out_queue.put(("opencode", None, str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def _opencode_planned(self, plan, err):
        if plan is None:
            self.set_status("Could not read opencode's config.", "Warn.TLabel")
            self.log("opencode: %s" % err, "err")
            return
        self.set_status("")
        if not plan.edits:
            self._show_text("opencode reasoning", plan.summary(),
                            notes=[plan.path])
            return

        def apply():
            try:
                backup = opencode.write(plan)
            except Exception as exc:                  # noqa: BLE001
                self.log("opencode: could not write %s: %s" % (plan.path, exc),
                         "err")
                return
            self.log("opencode: updated %s (%d change%s). Previous version "
                     "saved as %s. Restart opencode to pick it up."
                     % (plan.path, len(plan.changed),
                        "" if len(plan.changed) == 1 else "s", backup))
            self.set_status("opencode config updated.", "Ok.TLabel")
        self._show_text("opencode reasoning", plan.summary(),
                        notes=["%s - only reasoning and variants change, and "
                               "the current file is backed up first."
                               % plan.path],
                        action=("Write changes", apply))

    def _load(self):
        if self.busy:
            return
        plan = self._build_plan()
        if plan is None:
            return
        self.current_plan = plan
        if plan.notes:
            self.view.set_notes("  ".join(plan.notes))
        model = self.selected_model()
        self.cfg["last_model"] = model.id if model else ""

        be = self.backend()
        clear_first = bool(self.one_at_a_time.get())

        def tag_existed():
            # Asked after any unloading, which can itself remove a tag: asked
            # before, a tag about to be deleted would be recorded as one that
            # already existed, and Zoomies would then refuse to remove it.
            pre = be.installed_names() if hasattr(be, "installed_names") else None
            return bool(pre) and plan.creates_tag in pre

        self.busy = True
        self.view.set_launch_enabled(False)
        self.set_status("Loading...")
        self.log("")
        self.log("=== %s ===" % os.path.basename(plan.script_path), "note")
        self.view.launch_started()
        self._cancel = threading.Event()
        self._launch_pid = 0
        threading.Thread(target=self._run_worker,
                         args=(plan, tag_existed, clear_first, self._cancel),
                         daemon=True).start()

    def _run_worker(self, plan, pre_existing, clear_first=False, cancel=None):
        """cancel: an Event the Cancel button sets - loads only. Downloads
        and unloads run to the end."""
        emit = lambda ln: self.out_queue.put(("line", ln))
        cancel = cancel or threading.Event()
        if clear_first:
            self._unload_others(emit)
        if callable(pre_existing):
            pre_existing = pre_existing()
        if cancel.is_set():
            res = runner.RunResult(False, -1, message=CANCELLED)
        elif plan.long_lived:
            res = self._spawn_server(plan, emit, cancel)
        else:
            res = runner.run_script(
                plan, on_line=emit,
                on_start=lambda pid: self._launch_started_pid(pid, plan, cancel))
        if cancel.is_set() and not res.ok:
            res = runner.RunResult(False, res.returncode, res.pid, res.lines,
                                   CANCELLED)
        self.out_queue.put(("done", plan, pre_existing, res))

    def _launch_started_pid(self, pid, plan, cancel):
        """The script behind a load is running. Called on the worker."""
        self._launch_pid = pid
        if cancel.is_set():              # Cancel came before there was a pid
            self._end_launch(pid, plan)

    def cancel_launch(self):
        """Stop a load part-way: the Cancel button while Loading."""
        cancel, plan = self._cancel, self.current_plan
        if cancel is None or cancel.is_set() or plan is None:
            return
        cancel.set()
        self.set_status("Cancelling...", "Warn.TLabel")
        self.log("[zoomies] cancelling the load", "note")
        pid = self._launch_pid
        if pid:
            threading.Thread(target=self._end_launch, args=(pid, plan),
                             daemon=True).start()

    def _end_launch(self, pid, plan):
        """End the script a load is running. For llama.cpp that is the whole
        tree - the server runs as the script's child - exactly as Unload
        stops it. For Ollama only the script itself: it may have started
        Ollama's own server, which must keep running."""
        if not state.alive_and_named(pid, "powershell"):
            return                        # already finished, or pid reused
        cmd = ["taskkill", "/F", "/PID", str(pid)]
        if plan.long_lived:
            cmd.insert(1, "/T")
        try:
            subprocess.run(cmd, capture_output=True, timeout=30,
                           creationflags=runner.CREATE_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as exc:
            self.out_queue.put(("line", "[zoomies] could not stop pid %d: %s"
                                % (pid, exc)))

    def _unload_others(self, emit):
        """One model at a time: unload everything reachable before loading.

        A fresh list, not the dashboard's copy, which can be two seconds old.
        Each unload is the same plan the Unload button runs, so Ollama's
        temporary tags are still removed and records still updated.
        """
        loaded = []
        for be in backends.REGISTRY.values():
            try:
                loaded.extend(be.list_loaded())
            except Exception:                         # noqa: BLE001
                pass
        for item in loaded:
            be = backends.get(item.backend)
            if be is None:
                continue
            if item.foreign_process:
                # Everything else here is unloaded by asking a manager that
                # stays running. This one would mean ending somebody else's
                # process, which is not something a checkbox should do behind
                # their back - Unload, or the Processes tab, asks first.
                emit("[zoomies] leaving %s alone: its server was started "
                     "outside Zoomies (%s)" % (item.label, item.endpoint))
                continue
            emit("[zoomies] one model at a time - unloading %s (%s)"
                 % (item.label, be.display_name))
            try:
                stop = be.build_stop(item, self.session)
                res = runner.run_script(stop, on_line=emit, timeout=180)
            except Exception as exc:                  # noqa: BLE001
                emit("[zoomies] could not unload %s: %s" % (item.label, exc))
                continue
            self.out_queue.put(("cleared", stop, res))

    def _spawn_server(self, plan, emit, cancel):
        """Start a server that is meant to outlive the script.

        Waiting for it to exit would block forever, so the script is spawned
        detached, its log is tailed for the user, and readiness is a TCP
        connect - never a string match on log output, because log wording
        changes between versions and a listening socket does not.
        """
        started = runner.spawn_script(plan)
        if not started.ok:
            return started
        self._launch_started_pid(started.pid, plan, cancel)
        stop = Either(self.shutdown, cancel)
        threading.Thread(
            target=runner.tail_log,
            args=(plan.log_path, emit, self.shutdown),
            daemon=True).start()
        emit("[zoomies] waiting for %s:%d ..." % (plan.host, plan.port))
        up = state.wait_for_port(plan.host, plan.port, timeout=900,
                                 cancel=stop)
        if cancel.is_set():
            return runner.RunResult(False, -1, started.pid, [], CANCELLED)
        if not up:
            return runner.RunResult(
                False, -1, started.pid, [],
                "the server never started listening on port %d" % plan.port)
        emit("[zoomies] %s is serving on port %d" % (plan.backend, plan.port))
        if plan.ready_check is not None:
            emit("[zoomies] waiting for the model to finish loading ...")
            while not stop.is_set():
                if plan.ready_check():
                    emit("[zoomies] model loaded")
                    break
                if not state.alive_and_named(started.pid, "powershell"):
                    if plan.ready_check():      # finished in the last moment
                        break
                    return runner.RunResult(
                        False, -1, started.pid, [],
                        "%s exited before the model loaded - see the log above"
                        % plan.backend)
                stop.wait(2.0)
        if cancel.is_set():
            return runner.RunResult(False, -1, started.pid, [], CANCELLED)
        return runner.RunResult(True, 0, started.pid)

    def _run_done(self, plan, pre_existing, res):
        self.busy = False
        self.view.set_launch_enabled(True)
        if plan.kind == "load":
            self._cancel = None
        if res.ok:
            self._record(plan, pre_existing, res)
            self.set_status("Done.", "Ok.TLabel")
        elif res.message == CANCELLED:
            self.set_status("Cancelled.", "Warn.TLabel")
            self.log("[zoomies] load cancelled", "note")
            if plan.creates_tag:
                # The script may have got as far as creating its temporary
                # tag. Recorded, it is cleaned up like any other; a tag that
                # never got made is forgotten at the next start.
                backends.record_created_tag(
                    self.session, plan.creates_tag,
                    plan.creates_tag[:-len(state.TAG_SUFFIX)], pre_existing)
                state.save_session(self.session)
            if not plan.long_lived:
                self.log("[zoomies] Ollama may still finish loading it in the "
                         "background; unload it if it shows up.", "note")
        else:
            detail = res.message or "exit code %s" % res.returncode
            self.set_status("Failed - %s" % detail, "Bad.TLabel")
            self.log("FAILED: %s" % detail, "err")
        if plan.kind == "load":
            self.view.launch_finished(res.ok)
        self._poll_once()
        self._refresh_processes()

    def _record(self, plan, pre_existing, res):
        """Session bookkeeping for a plan that succeeded."""
        loads = self.session.setdefault("zoomies_loads", {})
        if plan.kind == "load" and plan.model_id:
            loads[plan.backend] = plan.model_id
        elif plan.kind == "stop":
            loads.pop(plan.backend, None)
        be = backends.get(plan.backend)
        if plan.long_lived and be is not None:
            # Enough to find this server again after a restart, and to
            # stop the right process. The plan says which model it brought
            # up; the dropdown only says what is selected now.
            be.remember_server(self.session, plan, res.pid)
        elif plan.kind == "stop" and be is not None:
            be.forget_server(self.session, plan)
        if plan.kind == "download":
            self._reload_models()
        if plan.creates_tag:
            backends.record_created_tag(self.session, plan.creates_tag,
                                        plan.creates_tag[:-len(state.TAG_SUFFIX)],
                                        pre_existing)
        if plan.removes_tag:
            backends.forget_tag(self.session, plan.removes_tag)
        state.save_session(self.session)

    # ------------------------------------------------------------------
    # processes
    # ------------------------------------------------------------------

    def _refresh_processes(self):
        """Read the process table on a worker; a PowerShell round trip takes
        a second or two."""
        if self._procs_busy or self.shutdown.is_set():
            return
        self._procs_busy = True

        def work():
            try:
                items = processes.listing()
            except Exception:                         # noqa: BLE001
                items = None
            self.out_queue.put(("procs", items))
        threading.Thread(target=work, daemon=True).start()

    def _procs_ready(self, items):
        self._procs_busy = False
        if items is not None:
            self.procs = items
        self.view.show_procs(items)

    def _describe_procs(self, chosen):
        return "\n".join("  %s  (%s, pid %d)" % (p.what, processes.fmt_ram(p.ram), p.pid)
                         for p in chosen)

    def _clean_leftovers(self):
        left = [p for p in self.procs if p.leftover]
        if not left:
            self.set_status("No leftovers to clean up.", "Ok.TLabel")
            return
        if not messagebox.askyesno(
                "Clean up leftovers?",
                "End these processes?\n\n%s\n\nA chat still open in a terminal "
                "will be closed." % self._describe_procs(left), parent=self.root):
            return
        self._end_procs(left)

    def _end_selected(self):
        chosen = self.view.selected_procs()
        if not chosen:
            self.set_status("Select a row in Processes first.", "Warn.TLabel")
            return
        kept = [p for p in chosen if p.protected]
        chosen = [p for p in chosen if not p.protected]
        if kept:
            self.log("Not ending %s - Ollama's own server and tray app are left "
                     "alone." % ", ".join(p.what for p in kept), "note")
        if not chosen:
            return
        if not messagebox.askyesno(
                "End selected?",
                "End these processes?\n\n%s" % self._describe_procs(chosen),
                parent=self.root):
            return
        self._end_procs(chosen)

    def _end_procs(self, chosen):
        self.log("")
        self.log("=== ending %d process%s ===" % (len(chosen),
                                                "" if len(chosen) == 1 else "es"), "note")

        def work():
            for p in chosen:
                ok, message = processes.end(p, self.session)
                self.out_queue.put(("line", "[zoomies] %s (pid %d): %s"
                                    % (p.what, p.pid, message)))
            self.out_queue.put(("procs_ended", None))
        threading.Thread(target=work, daemon=True).start()

    def _unload_selected(self):
        chosen = self.view.selected_loaded()
        if not chosen:
            self.set_status("Select a row in Loaded first.", "Warn.TLabel")
            return
        for item in chosen:
            self._unload_one(item)

    def _unload_all(self):
        with self.lock:
            loaded = list(self.shared["loaded"])
        if not loaded:
            return
        outside = [i.label for i in loaded if i.foreign_process]
        question = "Unload %d model(s)?" % len(loaded)
        if outside:
            # The per-model question is skipped below, so this one has to
            # carry what would otherwise have been asked about each.
            question += ("\n\nThis ends the server holding %s, which Zoomies "
                         "did not start." % ", ".join(outside))
        if not messagebox.askyesno("Unload everything?", question,
                                   parent=self.root):
            return
        for item in loaded:
            self._unload_one(item, confirm=False)

    def _unload_one(self, loaded, confirm=True):
        be = backends.get(loaded.backend)
        if be is None:
            return
        if confirm and not loaded.owned_by_us:
            question = "%s was not started by Zoomies.\n\n" % loaded.label
            question += ("Ending its server is the only way to free the VRAM. "
                         "Stop it?" if loaded.foreign_process
                         else "Unload it anyway?")
            if not messagebox.askyesno("Not started by Zoomies", question,
                                       parent=self.root):
                return
        plan = be.build_stop(loaded, self.session)
        self.current_plan = plan
        self.log("")
        self.log("=== unload %s ===" % loaded.label, "note")
        for note in plan.notes:
            self.log("  " + note, "note")
        self.busy = True
        threading.Thread(target=self._run_worker, args=(plan, False),
                         daemon=True).start()

    # ------------------------------------------------------------------
    # temporary tag management
    # ------------------------------------------------------------------

    def _known_tags(self):
        return [r.get("tag") for r in
                self.session.get("ollama", {}).get("created_tags", [])
                if r.get("tag")]

    def _startup_checks(self):
        ok, why = runner.powershell_available()
        if not ok:
            self.log("PowerShell is not usable: %s" % why, "err")
            messagebox.showerror(
                "PowerShell blocked",
                "Zoomies runs its launch scripts through PowerShell, and that "
                "is not working here:\n\n%s\n\nLoading models will fail until "
                "this is resolved." % why, parent=self.root)

        be = backends.get("ollama")
        known = self._known_tags()
        if not known or not be.server_up():
            return
        installed = set(be.installed_names() or ())
        loaded = {m.id for m in be.list_loaded()}
        orphans = []
        for tag in known:
            if tag not in installed:
                backends.forget_tag(self.session, tag)
                continue
            allowed, _ = backends.can_remove_tag(tag, self.session, loaded)
            if allowed:
                orphans.append(tag)
        state.save_session(self.session)
        if not orphans:
            return
        if messagebox.askyesno(
                "Leftover temporary tags",
                "These temporary tags were left behind by a previous run:\n\n"
                "%s\n\nRemove them now? (The base models are not touched.)"
                % "\n".join("  " + t for t in orphans), parent=self.root):
            self._sweep(orphans)

    def _sweep(self, tags):
        be = backends.get("ollama")
        plan = be.build_sweep(tags)
        self.current_plan = plan
        self.log("=== removing leftover tags ===", "note")
        res = runner.run_script(plan, on_line=lambda ln: self.out_queue.put(("line", ln)))
        for tag in tags:
            backends.forget_tag(self.session, tag)
        state.save_session(self.session)
        if not res.ok:
            self.log("sweep failed: %s" % res.message, "err")

    def _manage_tags(self):
        be = backends.get("ollama")
        installed = set(be.installed_names() or ())
        loaded = {m.id for m in be.list_loaded()}
        rows = []
        for tag in self._known_tags():
            if tag not in installed:
                continue
            allowed, reason = backends.can_remove_tag(tag, self.session, loaded)
            rows.append((tag, "removable" if allowed else reason))
        if not rows:
            messagebox.showinfo(
                "Temporary tags",
                "Zoomies has not left any temporary tags behind.\n\n"
                "They are created when you load a model with settings and "
                "deleted again when you unload it.", parent=self.root)
            return
        removable = [t for t, s in rows if s == "removable"]
        body = "\n".join("  %s   [%s]" % (t, s) for t, s in rows)
        if removable and messagebox.askyesno(
                "Temporary tags",
                "%s\n\nRemove the %d removable tag(s) now?"
                % (body, len(removable)), parent=self.root):
            self._sweep(removable)
        elif not removable:
            messagebox.showinfo("Temporary tags", body, parent=self.root)

    # ------------------------------------------------------------------
    # polling + refresh
    # ------------------------------------------------------------------

    def _poll_once(self):
        loaded, status = [], {}
        for name, be in backends.REGISTRY.items():
            try:
                available, why = be.is_available()
                up = bool(be.port) and state.port_open(be.host, be.port)
                status[name] = (available, why, up)
                loaded.extend(be.list_loaded())
            except Exception as exc:                  # noqa: BLE001
                status[name] = (False, str(exc), False)
        with self.lock:
            self.shared["loaded"] = loaded
            self.shared["status"] = status
            self.shared["polled"] = True
        self._sync_opencode_limits(loaded)

    def _sync_opencode_limits(self, loaded):
        """Keep opencode's idea of the context in step with the servers.

        opencode reads the window size out of its own config and never asks
        the server, so a number left over from an earlier load decides when
        it compacts. Hooked onto the poll rather than onto the load button
        this covers every way a model can come up, including a server
        started outside Zoomies, and costs a set comparison when nothing has
        changed.

        The servers are half of what can go out of step: the config itself
        is edited by hand, by the full sync, and by us, so its stamp is in
        the comparison too. Without it a number that went wrong while the
        same server stayed up - or a write that could not happen because the
        file was mid-edit - would stand until something was loaded or
        unloaded.
        """
        servers = tuple(sorted(
            (model.endpoint, model.id, model.context) for model in loaded
            if model.backend == "llamacpp" and model.context))
        signature = (servers, opencode.config_stamp())
        if signature == self._opencode_seen:
            return
        self._opencode_seen = signature
        if not servers:
            return
        try:
            plan = opencode.plan_limits(loaded=loaded)
            if plan.changed:
                opencode.write(plan, backup=False)
                for line in plan.changed:
                    self.out_queue.put(("line", "[zoomies] opencode " + line))
                # Our own write moved the stamp; record where we left it so
                # the next poll does not read the file back to find itself.
                self._opencode_seen = (servers, opencode.config_stamp())
            self._opencode_moan = ""
        except (OSError, ValueError) as exc:
            # A config that is missing, half-edited or not ours to parse is
            # not worth interrupting a load over. Said once, then left until
            # something changes rather than repeated at every load.
            if str(exc) != self._opencode_moan:
                self._opencode_moan = str(exc)
                self.out_queue.put(
                    ("line", "[zoomies] could not tell opencode the context: %s"
                     % exc))

    def _poll_loop(self):
        n = 0
        while not self.shutdown.is_set():
            self._poll_once()
            if n % 15 == 0:                 # every 30 s: keeps the leftover count current
                self._refresh_processes()
            n += 1
            self.shutdown.wait(POLL_SECONDS)

    def refresh(self):
        # drain worker messages
        try:
            while True:
                msg = self.out_queue.get_nowait()
                kind = msg[0]
                if kind == "line":
                    self.log(msg[1], "err" if runner.looks_like_error(msg[1]) else None)
                elif kind == "done":
                    self._run_done(msg[1], msg[2], msg[3])
                elif kind == "cleared":
                    if msg[2].ok:
                        self._record(msg[1], False, msg[2])
                        # Straight away, not at the next two-second poll: the
                        # model is gone, and a row that outlives it is the one
                        # thing on screen still claiming otherwise.
                        self._poll_once()
                    else:
                        self.log("could not unload before loading: %s"
                                 % (msg[2].message or "exit code %s" % msg[2].returncode),
                                 "err")
                elif kind == "procs":
                    self._procs_ready(msg[1])
                elif kind == "procs_ended":
                    state.save_session(self.session)
                    self._poll_once()
                    self._procs_busy = False
                    self._refresh_processes()
                elif kind == "optimal":
                    self._apply_result(msg[1], msg[2], keep_edits=msg[3])
                elif kind == "reasoning":
                    self._reasoning_ready(msg[1], msg[2])
                elif kind == "opencode":
                    self._opencode_planned(msg[1], msg[2])
                elif kind == "vram":
                    self._vram_ready(msg[1], msg[2])
                elif kind == "models":
                    self._models_ready(msg[1], msg[2], msg[3], msg[4])
        except queue.Empty:
            pass

        with self.lock:
            loaded = list(self.shared["loaded"])
            status = dict(self.shared["status"])
            polled = self.shared.get("polled", False)

        self.view.show_backend_status(status)
        self.loaded_items = loaded
        self.view.show_loaded(loaded)
        if polled and not self._first_poll_seen:
            self._first_poll_seen = True
            self.view.first_poll(loaded)

        if self.metrics is not None:
            snap = self.metrics.snapshot()
            self.view.show_live(snap)
            # Learned first: the spill itself is the measurement that says
            # what the card that ran out will hand out, and the advice the
            # alert gives is worked out against exactly that number.
            self._learn_vram(snap)
            self._check_spills(snap)
            self._vram_follow_live()

        self.after_id = self.root.after(REFRESH_MS, self.refresh)

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    def exit(self, unload=False):
        """The menu's Exit, and Unload all and exit - which unloads this
        once without changing the Unload on exit setting."""
        if unload:
            with self.lock:
                loaded = list(self.shared["loaded"])
            ours = [i.label for i in loaded if not i.foreign_process]
            outside = [i.label for i in loaded if i.foreign_process]
            if not ours:
                question = "Nothing Zoomies can unload is running.\n\nExit anyway?"
            else:
                question = "Unload %s and exit?" % ", ".join(ours)
            if outside:
                # Same rule as Unload on exit: servers somebody else started
                # are theirs to stop.
                question += ("\n\n%s stays running - its server was started "
                             "outside Zoomies." % ", ".join(outside))
            if not messagebox.askyesno("Unload all and exit", question,
                                       parent=self.root):
                return
            self._unload_now = True
        self._on_close()

    def _on_close(self):
        self.shutdown.set()
        self.view.close()
        for pending in (self.after_id, self._vram_after):
            if pending:
                try:
                    self.root.after_cancel(pending)
                except tk.TclError:
                    pass
        if self.metrics is not None:
            # stop(), not just cleanup(): the GPU thread may be halfway through
            # a typeperf sample. Exiting without killing it leaves typeperf
            # running and writing a sample file nobody will ever delete.
            self.metrics.stop()
        self.cfg["unload_on_exit"] = bool(self.unload_exit.get())
        self.cfg["one_model_at_a_time"] = bool(self.one_at_a_time.get())
        if self.view.remember_geometry and self.root.state() == "normal":
            self.cfg["geometry_" + self.layout_name] = self.root.geometry()
        state.save_config(self.cfg)
        state.save_session(self.session)

        # Loaded models are deliberately left running - keeping them warm is
        # the whole point of the tool. Opt in if you want otherwise. Never on
        # a layout switch, which only closes the window to reopen it.
        if (self.unload_exit.get() or self._unload_now) and not self._restarting:
            with self.lock:
                loaded = list(self.shared["loaded"])
            for item in loaded:
                be = backends.get(item.backend)
                # Servers somebody else started are left running: this setting
                # is about clearing up after Zoomies, not after everyone.
                if be is None or item.foreign_process:
                    continue
                try:
                    runner.run_script(be.build_stop(item, self.session), timeout=90)
                except Exception:                     # noqa: BLE001
                    pass
        self.root.destroy()


def relaunch():
    """Start the next Zoomies with the same interpreter and arguments, less
    any layout choice: config.json now says which one to open."""
    args, skip = [], False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg == "--layout":
            skip = True
            continue
        if arg.startswith("--layout=") or arg == "--classic":
            continue
        args.append(arg)
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args,
                         cwd=here)
    except OSError as exc:
        messagebox.showerror("Could not restart Zoomies",
                             "Start it again by hand - it will open in the "
                             "new layout.\n\n%s" % exc)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Zoomies")
    parser.add_argument("--layout", choices=sorted(LAYOUTS),
                        help="open in this layout (and remember it)")
    for name in sorted(LAYOUTS):
        parser.add_argument("--" + name, action="store_const", dest="layout",
                            const=name, help="same as --layout %s" % name)
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    enable_dpi_awareness()
    root = tk.Tk()
    app = Zoomies(root, layout=args.layout)
    try:
        root.mainloop()
    finally:
        app.shutdown.set()
        for worker in app.workers:
            worker.join(timeout=3)
    if app._restarting:
        relaunch()


if __name__ == "__main__":
    sys.exit(main())
