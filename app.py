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

import ctypes
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import backends
import metrics
import runner
import state

try:
    import optimizer
except ImportError:          # the optimizer phase has not landed yet
    optimizer = None

APP_TITLE = "Zoomies"

# Same palette as the Ollama Monitor, so the two tools look like siblings.
BG = "#1e1e1e"
BG_PANEL = "#252526"
BG_FIELD = "#2d2d30"
FG = "#e0e0e0"
FG_DIM = "#9a9a9a"
ACCENT = "#4fc1ff"
BORDER = "#3a3d41"
OK_GREEN = "#6ac47a"
WARN = "#e0b050"
BAD = "#e06c75"

FONT = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_MONO = ("Consolas", 9)

POLL_SECONDS = 2.0
REFRESH_MS = 200


def human_bytes(n):
    if not n:
        return "-"
    return "%.1f GB" % (n / 1024.0 ** 3)


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
    def __init__(self, root):
        self.root = root
        self.cfg = state.load_config()
        self.session = state.load_session()

        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.shared = {"loaded": [], "status": {}, "models": []}
        self.out_queue = queue.Queue()

        self.busy = False
        self.models = []
        self.current_plan = None
        self.after_id = None
        self.source_note = ""
        self.doc_line = ""
        self.scale = 1.0
        self.mode_keys = {}
        self.opt_result = None

        self.metrics = None
        self._build_style()
        self._build_widgets()

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

    def _build_style(self):
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG)

        dpi = window_dpi(self.root)
        self.scale = max(1.0, dpi / 96.0)
        # Tk sizes fonts in points, and "tk scaling" is pixels-per-point. Set
        # it from the real DPI and every point-sized font lands at the right
        # physical size. Raw pixel measurements still need px() by hand.
        self.root.tk.call("tk", "scaling", dpi / 72.0)

        want_w, want_h = self.px(980), self.px(800)
        max_w = int(self.root.winfo_screenwidth() * 0.92)
        max_h = int(self.root.winfo_screenheight() * 0.92)
        width, height = min(want_w, max_w), min(want_h, max_h)
        self.root.geometry("%dx%d+%d+%d" % (
            width, height,
            max(0, (self.root.winfo_screenwidth() - width) // 2),
            max(0, (self.root.winfo_screenheight() - height) // 3)))
        self.root.minsize(min(self.px(780), max_w), min(self.px(560), max_h))
        dark_titlebar(self.root)

        st = ttk.Style()
        st.theme_use("clam")        # the only built-in theme that lets us
                                    # recolour everything on Windows
        st.configure(".", background=BG, foreground=FG, font=FONT,
                     fieldbackground=BG_FIELD, bordercolor=BORDER)
        st.configure("TFrame", background=BG)
        st.configure("Panel.TFrame", background=BG_PANEL)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Dim.TLabel", background=BG, foreground=FG_DIM)
        st.configure("Head.TLabel", background=BG, foreground=ACCENT,
                     font=FONT_BOLD)
        st.configure("Off.TLabel", background=BG, foreground="#5a5a5a")
        st.configure("Warn.TLabel", background=BG, foreground=WARN)
        st.configure("Bad.TLabel", background=BG, foreground=BAD)
        st.configure("Ok.TLabel", background=BG, foreground=OK_GREEN)
        st.configure("TButton", background=BG_FIELD, foreground=FG,
                     bordercolor=BORDER, focuscolor=BG, padding=(10, 4))
        st.map("TButton",
               background=[("active", "#3e3e42"), ("disabled", "#2a2a2a")],
               foreground=[("disabled", "#666666")])
        st.configure("Go.TButton", background="#0e639c", foreground="#ffffff",
                     font=FONT_BOLD, padding=(16, 6))
        st.map("Go.TButton",
               background=[("active", "#1177bb"), ("disabled", "#2a2a2a")],
               foreground=[("disabled", "#666666")])
        st.configure("TRadiobutton", background=BG, foreground=FG)
        st.map("TRadiobutton", background=[("active", BG)],
               foreground=[("disabled", "#666666")])
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.map("TCheckbutton", background=[("active", BG)])
        st.configure("TEntry", fieldbackground=BG_FIELD, foreground=FG,
                     insertcolor=FG, bordercolor=BORDER, padding=(4, 2))
        st.map("TEntry",
               fieldbackground=[("disabled", "#242427")],
               foreground=[("disabled", "#5a5a5a")],
               bordercolor=[("disabled", "#303034")])
        st.configure("TCombobox", fieldbackground=BG_FIELD, background=BG_FIELD,
                     foreground=FG, arrowcolor=FG, bordercolor=BORDER)
        st.map("TCombobox", fieldbackground=[("readonly", BG_FIELD)],
               foreground=[("disabled", "#666666")])
        st.configure("Treeview", background=BG_PANEL, fieldbackground=BG_PANEL,
                     foreground=FG, bordercolor=BORDER,
                     rowheight=self.px(22))
        st.configure("Treeview.Heading", background=BG_FIELD, foreground=ACCENT,
                     font=FONT_BOLD)
        st.map("Treeview", background=[("selected", "#094771")],
               foreground=[("selected", "#ffffff")])
        st.configure("TLabelframe", background=BG, bordercolor=BORDER)
        st.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                     font=FONT_BOLD)
        st.configure("TSeparator", background=BORDER)
        st.configure("Zoom.Horizontal.TProgressbar",
                     background=ACCENT, troughcolor=BG_FIELD,
                     bordercolor=BORDER, lightcolor=ACCENT,
                     darkcolor=ACCENT, thickness=self.px(12))
        st.configure("TNotebook", background=BG, bordercolor=BORDER,
                     tabmargins=(2, 4, 2, 0))
        st.configure("TNotebook.Tab", background=BG_FIELD,
                     foreground=FG_DIM, padding=(14, 5))
        st.map("TNotebook.Tab",
               background=[("selected", BG_PANEL)],
               foreground=[("selected", ACCENT)])

        self.root.option_add("*TCombobox*Listbox.background", BG_FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#094771")

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 3}

        # ---- top bar -------------------------------------------------
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(top, text="Zoomies", style="Head.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(side="left")
        self.on_top = tk.BooleanVar(value=self.cfg.get("always_on_top", False))
        ttk.Checkbutton(top, text="Always on top", variable=self.on_top,
                        command=self._toggle_top).pack(side="right")
        self.unload_exit = tk.BooleanVar(value=self.cfg.get("unload_on_exit", False))
        ttk.Checkbutton(top, text="Unload on exit",
                        variable=self.unload_exit).pack(side="right", padx=(0, 12))

        # ---- backend + model ----------------------------------------
        pick = ttk.Frame(self.root)
        pick.pack(fill="x", **pad)
        pick.columnconfigure(1, weight=1)

        ttk.Label(pick, text="Backend").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(pick)
        row.grid(row=0, column=1, sticky="ew")
        self.backend_var = tk.StringVar(value=self.cfg.get("last_backend", "ollama"))
        self.backend_status = {}
        for name in ("ollama", "unsloth"):
            be = backends.get(name)
            rb = ttk.Radiobutton(
                row, text=(be.display_name if be else name.title()),
                value=name, variable=self.backend_var,
                command=self._on_backend_change)
            rb.pack(side="left", padx=(0, 14))
            if be is None:
                rb.state(["disabled"])
            lbl = ttk.Label(row, text="", style="Dim.TLabel")
            lbl.pack(side="left", padx=(0, 18))
            self.backend_status[name] = lbl

        ttk.Label(pick, text="Folder").grid(row=1, column=0, sticky="w", pady=3)
        frow = ttk.Frame(pick)
        frow.grid(row=1, column=1, sticky="ew", pady=3)
        frow.columnconfigure(0, weight=1)
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(frow, textvariable=self.folder_var)
        self.folder_entry.grid(row=0, column=0, sticky="ew")
        self.browse_btn = ttk.Button(frow, text="Browse...", command=self._browse)
        self.browse_btn.grid(row=0, column=1, padx=(6, 0))
        ttk.Button(frow, text="Open", command=self._open_folder).grid(
            row=0, column=2, padx=(6, 0))
        ttk.Button(frow, text="Rescan", command=self._reload_models).grid(
            row=0, column=3, padx=(6, 0))

        ttk.Label(pick, text="Model").grid(row=2, column=0, sticky="w", pady=3)
        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(pick, textvariable=self.model_var,
                                      state="readonly")
        self.model_box.grid(row=2, column=1, sticky="ew", pady=3)

        # ---- settings ------------------------------------------------
        box = ttk.LabelFrame(self.root, text=" Settings ")
        box.pack(fill="x", padx=8, pady=(8, 3))

        bar = ttk.Frame(box)
        bar.pack(fill="x", padx=8, pady=(6, 2))
        self.apply_btn = ttk.Button(bar, text="Apply optimal settings",
                                    command=self._apply_optimal)
        self.apply_btn.pack(side="left")
        if optimizer is None:
            self.apply_btn.state(["disabled"])
        ttk.Label(bar, text="Mode").pack(side="left", padx=(14, 4))
        self.mode_var = tk.StringVar()
        self.mode_box = ttk.Combobox(bar, textvariable=self.mode_var,
                                     state="readonly", width=22)
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", lambda e: self._mode_changed())
        ttk.Button(bar, text="Clear", command=self._clear_settings).pack(
            side="left", padx=(10, 0))
        # Answers are saved permanently once found, so there has to be a way
        # to go back and look again when the docs change.
        ttk.Button(bar, text="Re-read docs",
                   command=lambda: self._apply_optimal(force=True)).pack(
            side="left", padx=(6, 0))

        grid = ttk.Frame(box)
        grid.pack(fill="x", padx=8, pady=4)
        self.vars, self.entries, self.labels, self.dirty = {}, {}, {}, {}

        # Four label/entry column pairs. The entry columns share one uniform
        # group so every field is exactly the same width - without that, grid
        # hands the leftovers to whichever column has the longest label and
        # the row comes out ragged.
        for col in range(4):
            grid.columnconfigure(col * 2, weight=0)
            grid.columnconfigure(col * 2 + 1, weight=1, uniform="field")

        def make_field(key, row, col, span=1):
            lab = ttk.Label(grid, text=backends.SETTING_TEXT[key],
                            style="Dim.TLabel", anchor="e")
            lab.grid(row=row, column=col * 2, sticky="e", padx=(0, 6), pady=3)
            var = tk.StringVar()
            ent = ttk.Entry(grid, textvariable=var, justify="left")
            ent.grid(row=row, column=col * 2 + 1, sticky="ew",
                     padx=(0, self.px(18)), pady=3,
                     columnspan=(span * 2 - 1) if span > 1 else 1)
            var.trace_add("write", lambda *a, k=key: self._mark_dirty(k))
            self.vars[key], self.entries[key], self.labels[key] = var, ent, lab
            self.dirty[key] = False

        row = 0
        for row, keys in enumerate(backends.SETTING_ROWS):
            for col, key in enumerate(keys):
                if key:
                    make_field(key, row, col)
        for key in backends.SETTING_WIDE:
            row += 1
            make_field(key, row, 0, span=4)

        self.source_lbl = ttk.Label(box, text="", style="Dim.TLabel",
                                    wraplength=self.px(940), justify="left")
        self.source_lbl.pack(fill="x", padx=8, pady=(2, 0))
        self.notes_lbl = ttk.Label(box, text="", style="Warn.TLabel",
                                   wraplength=self.px(940), justify="left")
        self.notes_lbl.pack(fill="x", padx=8, pady=(2, 6))

        act = ttk.Frame(box)
        act.pack(fill="x", padx=8, pady=(0, 8))
        self.load_btn = ttk.Button(act, text="Load model", style="Go.TButton",
                                   command=self._load)
        self.load_btn.pack(side="left")
        ttk.Button(act, text="Preview script",
                   command=self._preview).pack(side="left", padx=(8, 0))
        self.status_lbl = ttk.Label(act, text="", style="Dim.TLabel")
        self.status_lbl.pack(side="left", padx=(16, 0))

        # ---- loaded --------------------------------------------------
        lbox = ttk.LabelFrame(self.root, text=" Loaded ")
        lbox.pack(fill="x", padx=8, pady=3)
        cols = ("model", "backend", "vram", "context", "endpoint", "pid", "until")
        widths = (300, 90, 80, 80, 150, 60, 80)
        self.tree = ttk.Treeview(lbox, columns=cols, show="headings", height=3)
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=self.px(w), minwidth=self.px(40),
                             anchor="w" if col in ("model", "endpoint") else "center")
        self.tree.pack(fill="x", padx=8, pady=(6, 2))
        self.tree.tag_configure("foreign", foreground=FG_DIM)
        self.tree.tag_configure("ours", foreground=FG)
        self.tree.bind("<Double-1>", lambda e: self._unload_selected())

        lbar = ttk.Frame(lbox)
        lbar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(lbar, text="Unload selected",
                   command=self._unload_selected).pack(side="left")
        ttk.Button(lbar, text="Unload all",
                   command=self._unload_all).pack(side="left", padx=(8, 0))
        ttk.Button(lbar, text="Temporary tags...",
                   command=self._manage_tags).pack(side="left", padx=(8, 0))
        self.vram_lbl = ttk.Label(lbar, text="", style="Dim.TLabel")
        self.vram_lbl.pack(side="right")

        self._build_live(self.root)

        self._on_backend_change(initial=True)
        self._toggle_top()

    def _build_live(self, parent):
        """Compact live-metrics strip plus a tabbed History/Output pane.

        Tabs rather than three stacked panes: the window is already tall, and
        History and Output are rarely both wanted at once.
        """
        box = ttk.LabelFrame(parent, text=" Live ")
        box.pack(fill="both", expand=True, padx=8, pady=3)

        top = ttk.Frame(box)
        top.pack(fill="x", padx=8, pady=(6, 2))
        self.live_status = ttk.Label(top, text="Waiting for activity...",
                                     style="Head.TLabel")
        self.live_status.pack(side="left")
        self.live_gpu = ttk.Label(top, text="", style="Dim.TLabel")
        self.live_gpu.pack(side="right")

        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(0, 2))
        self.live_stats = {}
        for key, label in (("prompt", "Prompt eval"), ("ttft", "TTFT"),
                           ("gen", "Generation"), ("tokens", "Generated")):
            cell = ttk.Frame(row)
            cell.pack(side="left", padx=(0, self.px(22)))
            ttk.Label(cell, text=label, style="Dim.TLabel").pack(anchor="w")
            value = ttk.Label(cell, text="-", style="TLabel",
                              font=("Consolas", 11, "bold"))
            value.pack(anchor="w")
            self.live_stats[key] = value

        ctx = ttk.Frame(box)
        ctx.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Label(ctx, text="Context", style="Dim.TLabel",
                  width=9).pack(side="left")
        self.ctx_bar = ttk.Progressbar(ctx, mode="determinate", maximum=1000,
                                       style="Zoom.Horizontal.TProgressbar")
        self.ctx_bar.pack(side="left", fill="x", expand=True)
        self.ctx_label = ttk.Label(ctx, text="-", style="Dim.TLabel")
        self.ctx_label.pack(side="left", padx=(8, 0))

        tabs = ttk.Notebook(box)
        tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        hist = ttk.Frame(tabs)
        tabs.add(hist, text="  History  ")
        cols = ("time", "backend", "model", "ttft", "tok/s", "tokens", "context")
        widths = (70, 80, 240, 70, 80, 80, 90)
        self.hist_tree = ttk.Treeview(hist, columns=cols, show="headings",
                                      height=7)
        for col, w in zip(cols, widths):
            self.hist_tree.heading(col, text=col.title())
            self.hist_tree.column(col, width=self.px(w), minwidth=self.px(40),
                                  anchor="w" if col == "model" else "center")
        self.hist_tree.pack(fill="both", expand=True)

        out = ttk.Frame(tabs)
        tabs.add(out, text="  Output  ")
        bar = ttk.Frame(out)
        bar.pack(fill="x", pady=(4, 2))
        ttk.Button(bar, text="Open log", command=self._open_log).pack(side="right")
        ttk.Button(bar, text="Open folder",
                   command=self._open_scripts).pack(side="right", padx=(0, 6))
        wrap = ttk.Frame(out)
        wrap.pack(fill="both", expand=True)
        self.out = tk.Text(wrap, height=6, bg=BG_PANEL, fg=FG, font=FONT_MONO,
                           relief="flat", wrap="none", insertbackground=FG,
                           highlightthickness=1, highlightbackground=BORDER)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.out.yview)
        self.out.configure(yscrollcommand=sb.set, state="disabled")
        self.out.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.out.tag_configure("err", foreground=BAD)
        self.out.tag_configure("note", foreground=ACCENT)
        self.tabs = tabs

    def _refresh_live(self):
        snap = self.metrics.snapshot()

        backend = snap.get("backend") or ""
        label = backend and (backends.get(backend).display_name
                             if backends.get(backend) else backend)
        status = snap.get("status") or ""
        self.live_status.configure(
            text="%s%s" % (status, "   -   %s" % label if label else ""),
            style="Ok.TLabel" if status == "Generating..." else "Head.TLabel")

        self.live_stats["prompt"].configure(
            text=metrics.fmt(snap.get("prompt_tps"), " t/s"))
        self.live_stats["ttft"].configure(
            text=metrics.fmt(snap.get("ttft"), "s", 2))
        gen = metrics.fmt(snap.get("tg"), " t/s")
        if snap.get("tg3s") is not None:
            gen += "  (3s %s)" % metrics.fmt(snap.get("tg3s"))
        self.live_stats["gen"].configure(text=gen)
        self.live_stats["tokens"].configure(
            text=metrics.fmt_int(snap.get("n_gen")))

        used, total = snap.get("n_tokens"), snap.get("n_ctx")
        if used and total:
            self.ctx_bar.configure(value=min(1000, int(1000.0 * used / total)))
            self.ctx_label.configure(text="%s / %s" % (format(used, ","),
                                                       format(total, ",")))
        else:
            self.ctx_bar.configure(value=0)
            self.ctx_label.configure(text="-")

        gpus = snap.get("gpus") or []
        if gpus:
            parts = []
            for gpu in gpus:
                pct = gpu.get("pct")
                parts.append("%s %s" % (gpu["name"],
                                        "-" if pct is None else "%.0f%%" % pct))
            self.live_gpu.configure(text="   ".join(parts), style="Dim.TLabel")
        elif snap.get("gpu_error"):
            self.live_gpu.configure(text="GPU: %s" % snap["gpu_error"],
                                    style="Bad.TLabel")

        rows = snap.get("history") or []
        existing = set(self.hist_tree.get_children(""))
        wanted = set()
        for i, row in enumerate(rows):
            iid = "h%d" % i
            wanted.add(iid)
            be = backends.get(row.get("backend") or "")
            values = (
                row.get("time", ""),
                be.display_name if be else (row.get("backend") or "-"),
                row.get("model") or "-",
                metrics.fmt(row.get("ttft"), "s", 2),
                metrics.fmt(row.get("tg3s")),
                metrics.fmt_int(row.get("n_gen")),
                metrics.fmt_int(row.get("n_ctx")),
            )
            if iid in existing:
                self.hist_tree.item(iid, values=values)
            else:
                self.hist_tree.insert("", "end", iid=iid, values=values)
        for iid in existing - wanted:
            self.hist_tree.delete(iid)

    def _live_model_name(self, backend):
        """What to label a finished request with.

        Whatever that backend reports as loaded right now is the best answer
        available - llama.cpp's timing lines carry no model name at all.
        """
        try:
            be = backends.get(backend)
            loaded = be.list_loaded() if be else []
            if loaded:
                return loaded[0].label
        except Exception:                          # noqa: BLE001
            pass
        model = self.selected_model()
        return model.label if model else ""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def backend(self):
        return backends.get(self.backend_var.get()) or backends.get("ollama")

    def selected_model(self):
        idx = self.model_box.current()
        if idx < 0 or idx >= len(self.models):
            return None
        return self.models[idx]

    def log(self, text, tag=None):
        self.out.configure(state="normal")
        self.out.insert("end", text + "\n", tag or ())
        self.out.see("end")
        self.out.configure(state="disabled")

    def set_status(self, text, style="Dim.TLabel"):
        self.status_lbl.configure(text=text, style=style)

    def settings_dict(self):
        return {k: v.get().strip() for k, v in self.vars.items()}

    def _mark_dirty(self, key):
        self.dirty[key] = True
        try:
            self.entries[key].configure(foreground=FG)
        except tk.TclError:
            pass

    def _set_value(self, key, value, auto=True):
        """Auto-filled values render in accent blue so it is obvious at a
        glance which numbers came from the docs and which you typed."""
        var = self.vars[key]
        for mode, cbname in var.trace_info():      # detach so setting the value
            var.trace_remove(mode, cbname)         # does not mark it dirty
        var.set("" if value is None else str(value))
        var.trace_add("write", lambda *a, k=key: self._mark_dirty(k))
        self.dirty[key] = not auto
        try:
            self.entries[key].configure(foreground=ACCENT if auto else FG)
        except tk.TclError:
            pass

    def _clear_settings(self):
        for key in self.vars:
            self._set_value(key, "", auto=False)
            self.dirty[key] = False
        self.source_lbl.configure(text="")
        self.notes_lbl.configure(text="")
        self.mode_box.configure(values=())
        self.mode_var.set("")
        self.mode_keys = {}
        self.opt_result = None
        self.source_note = ""

    # ------------------------------------------------------------------
    # backend / model wiring
    # ------------------------------------------------------------------

    def _on_backend_change(self, initial=False):
        be = self.backend()
        self.cfg["last_backend"] = be.name

        if be.uses_model_folder:
            self.folder_entry.state(["!disabled"])
            self.browse_btn.state(["!disabled"])
            self.folder_var.set(self.cfg.get("unsloth_folder")
                                or be.default_folder or "")
        else:
            self.folder_var.set("%s   (%s manages these)"
                                % (be.default_folder, be.display_name))
            self.folder_entry.state(["disabled"])
            self.browse_btn.state(["disabled"])

        for key, ent in self.entries.items():
            reason = be.supports(key)
            if reason:
                ent.state(["!disabled"])
                self.labels[key].configure(style="Dim.TLabel")
            else:
                ent.state(["disabled"])
                self.labels[key].configure(style="Off.TLabel")

        unsupported = [lab for key, lab in backends.SETTING_LABELS
                       if not be.supports(key)]
        msg = ""
        if unsupported:
            msg = "%s ignores: %s." % (be.display_name, ", ".join(unsupported))
        self.notes_lbl.configure(text=msg)

        self._reload_models()

    def _reload_models(self):
        be = self.backend()
        folder = self.folder_var.get() if be.uses_model_folder else None
        self.models = be.list_models(folder)
        labels = [m.describe() for m in self.models]
        self.model_box.configure(values=labels)
        want = self.cfg.get("last_model", "")
        chosen = 0
        for i, m in enumerate(self.models):
            if m.id == want:
                chosen = i
                break
        if labels:
            self.model_box.current(chosen)
        else:
            self.model_var.set("")
        if be.last_error and not labels:
            self.log("%s: %s" % (be.display_name, be.last_error), "err")

    def _browse(self):
        chosen = filedialog.askdirectory(
            title="Pick the folder holding your model files",
            initialdir=self.folder_var.get() or os.path.expanduser("~"))
        if chosen:
            self.folder_var.set(chosen)
            self.cfg["unsloth_folder"] = chosen
            self._reload_models()

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
        self.apply_btn.state(["disabled"])
        threading.Thread(target=self._apply_worker, args=(model, force),
                         daemon=True).start()

    def _apply_worker(self, model, force=False):
        try:
            result = optimizer.recommend(model, self.cfg,
                                         force_refresh=force)
        except Exception as exc:                      # noqa: BLE001
            result = optimizer.Result(error=str(exc))
        self.out_queue.put(("optimal", model, result))

    def _apply_result(self, model, result):
        self.apply_btn.state(["!disabled"])
        be = self.backend()
        self.opt_result = result

        if not result.settings:
            self.source_lbl.configure(text=result.error or
                                      "Nothing usable found on that page.")
            self.set_status("No settings found - type them in, or pick a page.",
                            "Warn.TLabel")
            if result.candidates:
                self._offer_page_picker(model, result.candidates)
            return

        applied, skipped = [], []
        for key, value in result.settings.items():
            if key not in self.vars:
                continue
            if be.supports(key):
                self._set_value(key, value, auto=True)
                applied.append(key)
            else:
                skipped.append(backends.SETTING_TEXT.get(key, key))

        # The dropdown shows the readable label; the lookup maps it back to
        # the internal key so flipping modes re-reads the right column.
        self.mode_keys = {result.mode_labels.get(k, k): k for k in result.modes}
        self.mode_box.configure(values=tuple(self.mode_keys))
        if result.mode:
            self.mode_var.set(result.mode_labels.get(result.mode, result.mode))

        self.source_note = result.source_line()
        self.source_lbl.configure(text=result.describe())
        msg = "Applied %d setting%s." % (len(applied),
                                         "" if len(applied) == 1 else "s")
        if skipped:
            msg += "  %s cannot use: %s." % (be.display_name, ", ".join(skipped))
        self.set_status(msg, "Ok.TLabel")
        for hint in result.suggestions:
            self.log("Not applied: " + hint, "note")

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
                 optimizer.recommend(model, self.cfg, mode=key))),
            daemon=True).start()

    def _offer_page_picker(self, model, candidates):
        """The docs have no page under this model's name.

        Rather than guessing a near-match - which is how a model quietly ends
        up running another model's sampling settings - ask.
        """
        win = tk.Toplevel(self.root)
        win.title("Which docs page?")
        win.configure(bg=BG)
        win.geometry("%dx%d" % (self.px(620), self.px(430)))
        win.transient(self.root)
        ttk.Label(win, wraplength=self.px(580), justify="left",
                  style="Warn.TLabel",
                  text=('There is no Unsloth docs page named "%s", so nothing '
                        "was filled in.\n\nIf you know which page applies, pick "
                        "it here and Zoomies will remember the choice for this "
                        "model. Leave it alone if you are not sure - a wrong "
                        "page is worse than an empty box."
                        % model.match_key)).pack(fill="x", padx=12, pady=(12, 8))

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

    def _show_text(self, title, body, notes=()):
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
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))

    def _load(self):
        if self.busy:
            return
        plan = self._build_plan()
        if plan is None:
            return
        self.current_plan = plan
        if plan.notes:
            self.notes_lbl.configure(text="  ".join(plan.notes))
        model = self.selected_model()
        self.cfg["last_model"] = model.id if model else ""

        pre = self.backend().installed_names() if hasattr(
            self.backend(), "installed_names") else None
        pre_existing = bool(pre) and plan.creates_tag in pre

        self.busy = True
        self.load_btn.state(["disabled"])
        self.set_status("Loading...")
        self.log("")
        self.log("=== %s ===" % os.path.basename(plan.script_path), "note")
        threading.Thread(target=self._run_worker,
                         args=(plan, pre_existing), daemon=True).start()

    def _run_worker(self, plan, pre_existing):
        emit = lambda ln: self.out_queue.put(("line", ln))
        if plan.long_lived:
            res = self._spawn_server(plan, emit)
        else:
            res = runner.run_script(plan, on_line=emit)
        self.out_queue.put(("done", plan, pre_existing, res))

    def _spawn_server(self, plan, emit):
        """Start a server that is meant to outlive the script.

        Waiting for it to exit would block forever, so the script is spawned
        detached, its log is tailed for the user, and readiness is a TCP
        connect - never a string match on log output, because log wording
        changes between versions and a listening socket does not.
        """
        started = runner.spawn_script(plan)
        if not started.ok:
            return started
        threading.Thread(
            target=runner.tail_log,
            args=(plan.log_path, emit, self.shutdown),
            daemon=True).start()
        emit("[zoomies] waiting for %s:%d ..." % (plan.host, plan.port))
        up = state.wait_for_port(plan.host, plan.port, timeout=900,
                                 cancel=self.shutdown)
        if not up:
            return runner.RunResult(
                False, -1, started.pid, [],
                "the server never started listening on port %d" % plan.port)
        emit("[zoomies] %s is serving on port %d" % (plan.backend, plan.port))
        return runner.RunResult(True, 0, started.pid)

    def _run_done(self, plan, pre_existing, res):
        self.busy = False
        self.load_btn.state(["!disabled"])
        if res.ok:
            if plan.long_lived:
                # Record enough to find this server again after a restart,
                # and to kill it properly: the server spawns llama-server.exe
                # as a child, so the parent pid alone is not enough.
                model = self.selected_model()
                self.session["unsloth"] = {
                    "shell_pid": res.pid,
                    "pid": (state.find_processes("unsloth.exe") or [0])[-1],
                    "port": plan.port, "host": plan.host,
                    "model_id": model.id if model else "",
                    "label": model.label if model else "",
                    "started": __import__("time").time(),
                    "log": plan.log_path, "script": plan.script_path,
                }
            if plan.creates_tag:
                backends.record_created_tag(self.session, plan.creates_tag,
                                            plan.creates_tag[:-len(state.TAG_SUFFIX)],
                                            pre_existing)
            if plan.removes_tag:
                backends.forget_tag(self.session, plan.removes_tag)
            state.save_session(self.session)
            self.set_status("Done.", "Ok.TLabel")
        else:
            detail = res.message or "exit code %s" % res.returncode
            self.set_status("Failed - %s" % detail, "Bad.TLabel")
            self.log("FAILED: %s" % detail, "err")
        self._poll_once()

    def _unload_selected(self):
        rows = self.tree.selection()
        if not rows:
            self.set_status("Select a row in Loaded first.", "Warn.TLabel")
            return
        with self.lock:
            loaded = list(self.shared["loaded"])
        for row in rows:
            idx = int(row)
            if 0 <= idx < len(loaded):
                self._unload_one(loaded[idx])

    def _unload_all(self):
        with self.lock:
            loaded = list(self.shared["loaded"])
        if not loaded:
            return
        if not messagebox.askyesno(
                "Unload everything?",
                "Unload %d model(s)?" % len(loaded), parent=self.root):
            return
        for item in loaded:
            self._unload_one(item, confirm=False)

    def _unload_one(self, loaded, confirm=True):
        be = backends.get(loaded.backend)
        if be is None:
            return
        if confirm and not loaded.owned_by_us and not messagebox.askyesno(
                "Not started by Zoomies",
                "%s was not started by Zoomies.\n\nUnload it anyway?"
                % loaded.label, parent=self.root):
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

    def _poll_loop(self):
        while not self.shutdown.is_set():
            self._poll_once()
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
                elif kind == "optimal":
                    self._apply_result(msg[1], msg[2])
        except queue.Empty:
            pass

        with self.lock:
            loaded = list(self.shared["loaded"])
            status = dict(self.shared["status"])

        for name, lbl in self.backend_status.items():
            info = status.get(name)
            if not info:
                lbl.configure(text="", style="Dim.TLabel")
                continue
            available, why, up = info
            if up:
                lbl.configure(text="* running", style="Ok.TLabel")
            elif available:
                lbl.configure(text="o " + why, style="Dim.TLabel")
            else:
                lbl.configure(text="x " + why, style="Bad.TLabel")

        existing = set(self.tree.get_children(""))
        wanted = set()
        for i, item in enumerate(loaded):
            iid = str(i)
            wanted.add(iid)
            values = (
                item.label,
                backends.get(item.backend).display_name if backends.get(item.backend)
                else item.backend,
                human_bytes(item.vram_bytes),
                format(item.context, ",") if item.context else "-",
                item.endpoint.replace("http://", "") or "-",
                item.pid or "-",
                self._until(item.expires),
            )
            tag = "ours" if item.owned_by_us else "foreign"
            if iid in existing:
                self.tree.item(iid, values=values, tags=(tag,))
            else:
                self.tree.insert("", "end", iid=iid, values=values, tags=(tag,))
        for iid in existing - wanted:
            self.tree.delete(iid)

        if self.metrics is not None:
            self._refresh_live()

        total = sum(i.vram_bytes for i in loaded)
        self.vram_lbl.configure(
            text="VRAM in use: %s" % human_bytes(total) if total else "")

        self.after_id = self.root.after(REFRESH_MS, self.refresh)

    @staticmethod
    def _until(expires):
        if not expires:
            return "-"
        import datetime
        try:
            txt = expires.split(".")[0]
            when = datetime.datetime.fromisoformat(txt)
            secs = (when - datetime.datetime.now()).total_seconds()
            if secs <= 0:
                return "now"
            if secs < 3600:
                return "%dm" % round(secs / 60)
            return "%.1fh" % (secs / 3600)
        except (ValueError, TypeError):
            return "-"

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    def _on_close(self):
        self.shutdown.set()
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
        if self.metrics is not None:
            self.metrics.cleanup()
        self.cfg["unload_on_exit"] = bool(self.unload_exit.get())
        state.save_config(self.cfg)
        state.save_session(self.session)

        # Loaded models are deliberately left running - keeping them warm is
        # the whole point of the tool. Opt in if you want otherwise.
        if self.unload_exit.get():
            with self.lock:
                loaded = list(self.shared["loaded"])
            for item in loaded:
                be = backends.get(item.backend)
                if be is None:
                    continue
                try:
                    runner.run_script(be.build_stop(item, self.session), timeout=90)
                except Exception:                     # noqa: BLE001
                    pass
        self.root.destroy()


def main():
    enable_dpi_awareness()
    root = tk.Tk()
    app = Zoomies(root)
    try:
        root.mainloop()
    finally:
        app.shutdown.set()
        for worker in app.workers:
            worker.join(timeout=3)


if __name__ == "__main__":
    sys.exit(main())
