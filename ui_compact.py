r"""
Zoomies - the compact layout: a slim window meant to sit docked beside
whatever you are working in.

One view at a time - Setup, Monitor, History, Processes - picked from the
icons along the bottom (or Ctrl+1..4). Everything else the classic top bar
held lives in the ... menu. Launching jumps to Monitor; a failed launch
goes back to Setup, where it can be fixed.

Same contract as ui_classic: this module only draws. Values live on the
controller as tk variables, and buttons call controller methods.
"""

import collections
import csv
import datetime
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import backends
import metrics
import processes
import vram
from ui_common import (ACCENT, BAD, BG, BG_FIELD, BG_PANEL, BORDER, FG, FG_DIM,
                       FONT_MONO, OK_GREEN, WARN, human_bytes, until)

# Segoe Fluent Icons ships with Windows 11 and MDL2 Assets with Windows 10;
# both put the same glyphs at the same code points.
ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")
GLYPH = {
    "setup": "\ue9e9",        # Equalizer
    "monitor": "\ue9d9",      # Diagnostic
    "history": "\ue81c",      # History
    "procs": "\ue9f5",        # Processing
    "more": "\ue712",         # More
}
# Shown instead when neither icon font is installed.
FALLBACK = {"setup": "=", "monitor": "~", "history": "#", "procs": "*",
            "more": "..."}

VIEWS = (("setup", "Setup"), ("monitor", "Monitor"), ("history", "History"),
         ("procs", "Procs"))
TITLES = {"setup": "Setup", "monitor": "Monitor", "history": "History",
          "procs": "Processes"}

NAV_BG = BG_FIELD
NAV_HOVER = "#37373b"
HINT_BG = "#12324a"               # accent tint behind the context hint

# Setup's sections. KV cache is a dropdown rather than a typed value but
# sits in the grid like one; the rest are backends.SETTING_LABELS keys.
SAMPLING = ("temperature", "top_p", "top_k", "min_p")
SIZING = ("context_length", "gpu_layers", "kv_cache", "parallel")
ADVANCED = ("repeat_penalty", "presence_penalty", "seed", "keep_alive")
TONE = {"Ok.TLabel": OK_GREEN, "Warn.TLabel": WARN, "Bad.TLabel": BAD,
        "Dim.TLabel": FG_DIM}


def fmt_secs(seconds):
    """33.9 s, or 2:05 past a minute."""
    if seconds is None:
        return None
    if seconds < 60:
        return "%.1f s" % seconds
    whole = int(round(seconds))
    return "%d:%02d" % (whole // 60, whole % 60)


def fmt_k(n):
    """Tokens in 1024s, the way context sizes are spoken about: 32,768 ->
    32k, 11,200 -> 10.9k."""
    if not n:
        return "0"
    k = n / 1024.0
    if k < 1:
        return str(n)
    return "%dk" % round(k) if abs(k - round(k)) < 0.05 else "%.1fk" % k


def fmt_tps(value):
    """38.2 t/s, but 1,149 t/s - a decimal on a prompt-eval rate is noise,
    and the line it sits on is narrow."""
    if value is None:
        return ""
    return ("%s t/s" % format(int(round(value)), ",") if value >= 100
            else "%.1f t/s" % value)


def join(*parts):
    return " \u00b7 ".join(p for p in parts if p)


def day_label(date):
    """Today / Yesterday / 12 Sep, for the group rows in History."""
    if not date:
        return "Earlier"
    try:
        when = datetime.date.fromisoformat(date)
    except ValueError:
        return date
    days = (datetime.date.today() - when).days
    if days == 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    return when.strftime("%d %b").lstrip("0")


def short_model(row):
    """qwen3.8:27b-q4_K_M on llama.cpp -> qwen3.8:27b-q4_K_M \u00b7 lcpp."""
    short = {"llamacpp": "lcpp", "ollama": "oll"}
    backend = row.get("backend") or ""
    return join(row.get("model") or "-", short.get(backend, backend))


def detail_lines(row):
    """What the row hides until it is opened. Context is not repeated - it
    is already in the row itself."""
    out = []
    if row.get("ttft") is not None:
        out.append("TTFT " + metrics.fmt_ttft(row.get("ttft"),
                                              row.get("ttft_upper")))
    prompt = join("%s tok" % metrics.fmt_int(row["prompt_n"])
                  if row.get("prompt_n") else "",
                  "+%s cached" % metrics.fmt_int(row["prompt_cached"])
                  if row.get("prompt_cached") else "",
                  fmt_tps(row.get("prompt_tps")))
    if prompt:
        out.append("Prompt " + prompt)
    output = join("%s tok" % metrics.fmt_int(row["n_gen"])
                  if row.get("n_gen") else "",
                  fmt_secs(row.get("gen_s")) or fmt_secs(row.get("runtime")))
    if output:
        out.append("Output " + output)
    if row.get("kv"):
        out.append("KV " + str(row["kv"]))
    return out


CSV_FIELDS = ("date", "time", "model", "backend", "gen_avg", "prompt_tps",
              "ttft", "runtime", "n_gen", "n_tokens", "n_ctx", "kv",
              "prompt_n", "prompt_cached", "prompt_s", "gen_s")


class CompactLayout:
    name = "compact"
    size = (360, 900)             # at 100% display scale
    min_size = (300, 420)
    remember_geometry = True      # it lives docked where you put it
    dock = "right"                # where it opens the first time

    def __init__(self, app):
        self.app = app
        self.root = app.root
        self.px = app.px
        families = set(tkfont.families(self.root))
        self.icon_font = next((f for f in ICON_FONTS if f in families), None)
        self.current = None
        self._user_chose = False          # a view was picked by hand
        self._launching = False
        self._loaded = []
        self._leftovers = 0
        self._generating_tps = None
        self._folder_editable = False
        self._can_download = False
        self.log_lines = collections.deque(maxlen=500)
        self.entries, self.labels = {}, {}
        self._wrapping = []               # labels that wrap to the form's width
        self._form_width = 0              # what the last resize measured
        self._backend_status = {}
        self._est = None
        self._vram_labels = {}
        self._chips_key = None            # which loaded models the chips show
        self._picked = None               # (backend, endpoint, label) chosen
        self._gpu_rows = []
        self._launch_t0 = 0.0
        self._tick_id = None              # the loading view's one-second tick
        self._hist_filter = None          # a model name, or None for all
        self._hist_key = None             # what the table currently shows
        self._hist_rows = []              # the rows behind it, filtered
        self.procs = []
        self._procs_key = None            # what the list currently shows
        self._proc_pick = None            # the row End was pressed on
        self._styles()
        self._build()
        self.show("setup")

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _glyph(self, key):
        return GLYPH[key] if self.icon_font else FALLBACK[key]

    def _styles(self):
        st = ttk.Style()
        st.configure("Section.TLabel", background=BG, foreground=FG,
                     font=("Segoe UI", 9, "bold"))
        # Processes reads as a list of small print: what each one is, then
        # its numbers underneath.
        st.configure("Small.TLabel", background=BG, foreground=FG,
                     font=("Segoe UI", 8))
        st.configure("SmallDim.TLabel", background=BG, foreground=FG_DIM,
                     font=("Segoe UI", 8))
        st.configure("SmallWarn.TLabel", background=BG, foreground=WARN,
                     font=("Segoe UI", 8))
        st.configure("SmallOff.TLabel", background=BG, foreground="#5a5a5a",
                     font=("Segoe UI", 8))
        st.configure("Small.TButton", font=("Segoe UI", 8),
                     padding=(self.px(6), 0))
        st.configure("Bar.TFrame", background=BG_PANEL)
        st.configure("Bar.TLabel", background=BG_PANEL, foreground=FG_DIM)
        st.configure("Hint.TFrame", background=HINT_BG)
        st.configure("Card.TFrame", background=BG_PANEL)
        st.configure("Card.TLabel", background=BG_PANEL, foreground=FG_DIM)
        st.configure("CardValue.TLabel", background=BG_PANEL, foreground=FG,
                     font=("Segoe UI", 13, "bold"))
        st.configure("Big.TLabel", background=BG_PANEL, foreground=FG,
                     font=("Segoe UI", 20, "bold"))
        # Smaller than the classic table: this one is read at a glance in a
        # narrow window, and more rows on screen beats bigger type.
        st.configure("Compact.Treeview", background=BG, fieldbackground=BG,
                     foreground=FG, bordercolor=BORDER, borderwidth=0,
                     font=("Segoe UI", 8), rowheight=self.px(19))
        st.configure("Compact.Treeview.Heading", background=BG,
                     foreground=FG_DIM, font=("Segoe UI", 8), relief="flat")
        st.map("Compact.Treeview.Heading", background=[("active", BG)])
        st.layout("Compact.Treeview", [("Compact.Treeview.treearea",
                                        {"sticky": "nswe"})])
        # clam draws these light grey; the classic layout keeps its own.
        st.configure("Dark.Vertical.TScrollbar", background=BG_FIELD,
                     troughcolor=BG, bordercolor=BG, arrowcolor=FG_DIM,
                     lightcolor=BG_FIELD, darkcolor=BG_FIELD, gripcount=0)
        st.map("Dark.Vertical.TScrollbar",
               background=[("active", "#3e3e42")])
        st.configure("Dark.Horizontal.TScale", background=FG_DIM,
                     troughcolor=BG_FIELD, bordercolor=BG_FIELD,
                     lightcolor=FG_DIM, darkcolor=FG_DIM)
        st.map("Dark.Horizontal.TScale", background=[("active", FG)])
        for name, colour in (("Ok", OK_GREEN), ("Warn", WARN), ("Bad", BAD),
                             ("Zoom", ACCENT)):
            st.configure("%s.Thin.Horizontal.TProgressbar" % name,
                         background=colour, troughcolor=BG_FIELD,
                         bordercolor=BG_FIELD, lightcolor=colour,
                         darkcolor=colour, thickness=self.px(6))

    def _build(self):
        head = ttk.Frame(self.root)
        head.pack(side="top", fill="x", padx=self.px(10), pady=(self.px(8), 0))
        self.title_lbl = ttk.Label(head, text="", style="Head.TLabel",
                                   font=("Segoe UI", 11, "bold"))
        self.title_lbl.pack(side="left")
        self.more_btn = tk.Label(head, text=self._glyph("more"), bg=BG, fg=FG,
                                 cursor="hand2", padx=self.px(6),
                                 font=(self.icon_font or "Segoe UI", 12))
        self.more_btn.pack(side="right")
        self.more_btn.bind("<Button-1>", lambda e: self._post_menu())
        ttk.Separator(self.root).pack(side="top", fill="x", pady=(self.px(6), 0))

        # Packed before the body so it keeps its height when the window is
        # short - the views give way, not the navigation.
        nav = tk.Frame(self.root, bg=NAV_BG)
        nav.pack(side="bottom", fill="x")
        self.nav = {}
        for col, (key, text) in enumerate(VIEWS):
            nav.columnconfigure(col, weight=1, uniform="nav")
            self.nav[key] = self._nav_item(nav, col, key, text)
        ttk.Separator(self.root).pack(side="bottom", fill="x")

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.frames = {}
        for key, _text in VIEWS:
            frame = ttk.Frame(body)
            frame.grid(row=0, column=0, sticky="nsew")
            self.frames[key] = frame
        self._build_setup(self.frames["setup"])
        self._build_monitor(self.frames["monitor"])
        self._build_history(self.frames["history"])
        self._build_procs(self.frames["procs"])

        self._build_menu()
        for n, (key, _text) in enumerate(VIEWS, start=1):
            self.root.bind("<Control-Key-%d>" % n,
                           lambda e, k=key: self.show(k, user=True))

    def _nav_item(self, parent, col, key, text):
        cell = tk.Frame(parent, bg=NAV_BG, cursor="hand2")
        cell.grid(row=0, column=col, sticky="nsew")
        icon = tk.Label(cell, text=self._glyph(key), bg=NAV_BG, fg=FG_DIM,
                        font=(self.icon_font or "Segoe UI", 13))
        icon.pack(pady=(self.px(6), 0))
        label = tk.Label(cell, text=text, bg=NAV_BG, fg=FG_DIM,
                         font=("Segoe UI", 8))
        label.pack(pady=(0, self.px(6)))
        parts = (cell, icon, label)
        for w in parts:
            w.bind("<Button-1>", lambda e, k=key: self.show(k, user=True))
            w.bind("<Enter>", lambda e, p=parts: [x.configure(bg=NAV_HOVER)
                                                  for x in p])
            w.bind("<Leave>", lambda e, p=parts: [x.configure(bg=NAV_BG)
                                                  for x in p])
        return {"icon": icon, "label": label, "text": text}

    def _build_menu(self):
        app = self.app
        m = tk.Menu(self.root, tearoff=0)
        m.add_checkbutton(label="Always on top", variable=app.on_top,
                          command=app._toggle_top)
        m.add_checkbutton(label="Unload on exit", variable=app.unload_exit)
        m.add_checkbutton(label="One model at a time",
                          variable=app.one_at_a_time)
        m.add_separator()
        m.add_command(label="Model folder...", command=app._browse)
        m.add_command(label="Open model folder", command=app._open_folder)
        m.add_command(label="Rescan models", command=app._reload_models)
        m.add_command(label="Download model...", command=app._download)
        m.add_command(label="Temporary tags...", command=app._manage_tags)
        m.add_command(label="Sync opencode...", command=app._sync_opencode)
        m.add_separator()
        m.add_command(label="Open log", command=app._open_log)
        m.add_command(label="Open scripts folder", command=app._open_scripts)
        others = app.other_layouts()
        if others:
            m.add_separator()
            for name in others:
                m.add_command(label="Switch to %s layout" % name,
                              command=lambda n=name: app.switch_layout(n))
        m.add_separator()
        m.add_command(label="Exit", command=app.exit)
        m.add_command(label="Unload all and exit",
                      command=lambda: app.exit(unload=True))
        self.menu = m

    def _post_menu(self):
        # Greyed where the backend has no use for them, as the classic
        # layout's buttons are.
        self.menu.entryconfigure("Model folder...", state="normal"
                                 if self._folder_editable else "disabled")
        self.menu.entryconfigure("Download model...", state="normal"
                                 if self._can_download else "disabled")
        x = self.more_btn.winfo_rootx() + self.more_btn.winfo_width()
        y = self.more_btn.winfo_rooty() + self.more_btn.winfo_height()
        try:
            self.menu.tk_popup(x, y)
        finally:
            self.menu.grab_release()

    # ---- views -----------------------------------------------------------
    # Placeholders until each view is built (phases 2-5). They show enough
    # to confirm the plumbing - model, status, what is loaded - and say
    # where to go meanwhile.

    def _placeholder(self, parent, text):
        pad = self.px(12)
        lbl = ttk.Label(parent, text=text, style="Dim.TLabel", justify="left")
        lbl.pack(fill="x", padx=pad, pady=(pad, self.px(6)))
        parent.bind("<Configure>", lambda e, l=lbl: l.configure(
            wraplength=max(self.px(120), e.width - 2 * pad)))
        return lbl

    def _line(self, parent, var=None, style="TLabel"):
        lbl = ttk.Label(parent, text="", textvariable=var, style=style,
                        justify="left")
        lbl.pack(fill="x", padx=self.px(12), pady=(0, self.px(4)))
        parent.bind("<Configure>", lambda e, l=lbl: l.configure(
            wraplength=max(self.px(120), e.width - self.px(24))), add="+")
        return lbl

    # ---- building blocks -------------------------------------------------

    def _scrolled(self, parent):
        """A frame that scrolls when the window is shorter than it. The
        scrollbar only shows when there is something to scroll to."""
        wrap = ttk.Frame(parent)
        wrap.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview,
                            style="Dark.Vertical.TScrollbar")
        inner = ttk.Frame(canvas)
        item = canvas.create_window(0, 0, window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)

        def fit(_event=None):
            canvas.configure(scrollregion=(0, 0, inner.winfo_reqwidth(),
                                           inner.winfo_reqheight()))
            needed = inner.winfo_reqheight() > canvas.winfo_height()
            if needed and not bar.winfo_manager():
                bar.pack(side="right", fill="y", before=canvas)
            elif not needed and bar.winfo_manager():
                bar.pack_forget()
                canvas.yview_moveto(0)

        def resize(event):
            canvas.itemconfigure(item, width=event.width)
            self._form_width = event.width
            for entry in list(self._wrapping):
                lbl, spare = entry
                try:
                    lbl.configure(wraplength=max(self.px(100),
                                                 event.width - spare))
                except tk.TclError:
                    self._wrapping.remove(entry)   # a rebuilt row; it is gone
            fit()

        def wheel(event):
            if bar.winfo_manager():
                canvas.yview_scroll(int(-event.delta / 120), "units")

        inner.bind("<Configure>", fit)
        canvas.bind("<Configure>", resize)
        # Only while the pointer is over the form: an open dropdown list
        # scrolls itself, and must not drag the form along with it.
        canvas.bind("<Enter>", lambda e: self.root.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda e: self.root.unbind_all("<MouseWheel>"))
        return inner

    def _wrap(self, lbl, spare):
        """Wrap lbl to the form's width, less spare pixels of padding. Rows
        built after the last resize take the width it saw."""
        self._wrapping.append((lbl, self.px(spare)))
        if self._form_width:
            lbl.configure(wraplength=max(self.px(100),
                                         self._form_width - self.px(spare)))
        return lbl

    def _section(self, parent, title):
        box = ttk.Frame(parent)
        box.pack(fill="x", padx=self.px(12), pady=(self.px(10), self.px(8)))
        head = ttk.Frame(box)
        head.pack(fill="x", pady=(0, self.px(2)))
        ttk.Label(head, text=title, style="Section.TLabel").pack(side="left")
        ttk.Separator(parent).pack(fill="x")
        return box, head

    def _link(self, parent, text, command, bg=BG):
        lbl = tk.Label(parent, text=text, bg=bg, fg=ACCENT, cursor="hand2",
                       font=("Segoe UI", 9))
        lbl.bind("<Button-1>", lambda e: command()
                 if str(lbl.cget("fg")) == ACCENT else None)
        return lbl

    def _toggle(self, parent, text, command):
        """A flat button that can show as selected - backends and presets."""
        lbl = tk.Label(parent, text=text, bg=BG_FIELD, fg=FG, cursor="hand2",
                       font=("Segoe UI", 9), padx=self.px(8), pady=self.px(3),
                       highlightthickness=1, highlightbackground=BORDER)
        lbl.bind("<Button-1>", lambda e: command())
        return lbl

    @staticmethod
    def _paint_toggle(lbl, selected):
        lbl.configure(fg=ACCENT if selected else FG,
                      highlightbackground=ACCENT if selected else BORDER,
                      highlightcolor=ACCENT if selected else BORDER)

    def _fields(self, parent, keys):
        """Two per row, each with its label above it."""
        app = self.app
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(self.px(2), 0))
        for col in (0, 1):
            grid.columnconfigure(col, weight=1, uniform="field")
        for i, key in enumerate(keys):
            row, col = divmod(i, 2)
            cell = ttk.Frame(grid)
            cell.grid(row=row, column=col, sticky="ew", pady=(0, self.px(6)),
                      padx=(0, self.px(8)) if col == 0 else (self.px(8), 0))
            if key == "kv_cache":
                self.kv_label = ttk.Label(cell, text="KV cache", style="Dim.TLabel")
                self.kv_label.pack(anchor="w")
                self.kv_box = ttk.Combobox(cell, textvariable=app.kv_var,
                                           state="readonly", width=6,
                                           values=backends.KV_CACHE_CHOICES)
                self.kv_box.pack(fill="x")
                self.kv_box.bind("<<ComboboxSelected>>",
                                 lambda e: app._kv_changed())
                continue
            lab = ttk.Label(cell, text=backends.SETTING_TEXT[key], style="Dim.TLabel")
            lab.pack(anchor="w")
            ent = ttk.Entry(cell, textvariable=app.vars[key], width=6)
            ent.pack(fill="x")
            self.entries[key], self.labels[key] = ent, lab
        return grid

    # ---- setup -----------------------------------------------------------

    def _build_setup(self, f):
        app = self.app
        self._build_launch_bar(f)
        form = self._scrolled(f)

        # Model
        box, _head = self._section(form, "Model")
        ttk.Label(box, text="Backend", style="Dim.TLabel").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.backend_btns = {}
        for col, name in enumerate(("llamacpp", "ollama")):
            row.columnconfigure(col, weight=1, uniform="be")
            be = backends.get(name)
            cell = ttk.Frame(row)
            cell.grid(row=0, column=col, sticky="ew",
                      padx=(0, self.px(4)) if col == 0 else (self.px(4), 0))
            btn = self._toggle(cell, be.display_name if be else name.title(),
                               lambda n=name: self._pick_backend(n))
            btn.pack(fill="x")
            status = ttk.Label(cell, text="", style="Dim.TLabel", anchor="center")
            status.pack(fill="x")
            self.backend_btns[name] = (btn, status)
        app.backend_var.trace_add("write", lambda *a: self._paint_backends())
        self._paint_backends()

        ttk.Label(box, text="Model", style="Dim.TLabel").pack(
            anchor="w", pady=(self.px(6), 0))
        self.model_box = ttk.Combobox(box, textvariable=app.model_var,
                                      state="readonly")
        self.model_box.pack(fill="x")
        self.model_box.bind("<<ComboboxSelected>>",
                            lambda e: app.model_picked(self.model_box.current()))
        info = ttk.Frame(box)
        info.pack(fill="x", pady=(self.px(2), 0))
        self.models_lbl = ttk.Label(info, text="", style="Dim.TLabel")
        self.models_lbl.pack(side="left")
        self._link(info, "Rescan", app._reload_models).pack(side="right")

        # Profile
        box, head = self._section(form, "Profile")
        self.apply_link = self._link(head, "Fill from docs", app._apply_optimal)
        self.apply_link.pack(side="right")
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="Preset", style="Dim.TLabel").pack(side="left")
        self._link(row, "Save current", app._save_preset).pack(side="right")
        self.chips = ttk.Frame(box)
        self.chips.pack(fill="x")
        self._chip_btns = {}
        app.preset_var.trace_add("write", lambda *a: self._paint_chips())
        self.show_presets(())
        pair = ttk.Frame(box)
        pair.pack(fill="x", pady=(self.px(6), 0))
        for col in (0, 1):
            pair.columnconfigure(col, weight=1, uniform="pair")
        for col, (text, var, attr, handler) in enumerate((
                ("Mode", app.mode_var, "mode_box", app._mode_changed),
                ("Reasoning", app.reason_var, "reason_box", app._reason_changed))):
            cell = ttk.Frame(pair)
            cell.grid(row=0, column=col, sticky="ew",
                      padx=(0, self.px(8)) if col == 0 else (self.px(8), 0))
            ttk.Label(cell, text=text, style="Dim.TLabel").pack(anchor="w")
            box_ = ttk.Combobox(cell, textvariable=var, state="readonly", width=6)
            box_.pack(fill="x")
            box_.bind("<<ComboboxSelected>>", lambda e, h=handler: h())
            setattr(self, attr, box_)
        self.source_lbl = self._wrap(ttk.Label(box, text="", style="Dim.TLabel",
                                               justify="left"), 24)
        self.source_lbl.pack(fill="x", pady=(self.px(6), 0))

        # Sampling
        box, _head = self._section(form, "Sampling")
        self._fields(box, SAMPLING)

        # Sizing, with what the load will need beside the numbers driving it
        box, _head = self._section(form, "Sizing")
        self._fields(box, SIZING)
        self._build_vram(box)

        # Advanced, folded away: rarely touched, and the form is long enough
        box, head = self._section(form, "")
        self.adv_head = self._link(head, "", self._toggle_advanced)
        self.adv_head.configure(fg=FG_DIM)
        self.adv_head.bind("<Button-1>", lambda e: self._toggle_advanced())
        self.adv_head.pack(side="left")
        self.adv_body = ttk.Frame(box)
        self._fields(self.adv_body, ADVANCED)
        lab = ttk.Label(self.adv_body, text=backends.SETTING_TEXT["extra_flags"],
                        style="Dim.TLabel")
        lab.pack(anchor="w")
        ent = ttk.Entry(self.adv_body, textvariable=app.vars["extra_flags"])
        ent.pack(fill="x")
        # --chat-template-file changes which template sets the levels.
        ent.bind("<FocusOut>", lambda e: app._refresh_reasoning())
        self.entries["extra_flags"], self.labels["extra_flags"] = ent, lab
        links = ttk.Frame(self.adv_body)
        links.pack(fill="x", pady=(self.px(8), 0))
        # Answers are saved permanently once found, so there has to be a way
        # to go back and look again when the docs change.
        self._link(links, "Re-read docs",
                   lambda: app._apply_optimal(force=True)).pack(side="left")
        self._link(links, "Clear all", app._clear_settings).pack(side="right")
        self._advanced_open = False
        self._toggle_advanced(False)

        self.notes_lbl = self._wrap(ttk.Label(form, text="", style="Warn.TLabel",
                                              justify="left"), 24)
        self.notes_lbl.pack(fill="x", padx=self.px(12), pady=(self.px(8), self.px(12)))

    def _build_vram(self, box):
        """Per card: what the load will need, before loading it."""
        app = self.app
        self.vram_frame = ttk.Frame(box)
        self.vram_frame.pack(fill="x", pady=(self.px(2), 0))
        self.vram_bars = ttk.Frame(self.vram_frame)
        self.vram_bars.pack(fill="x")
        self.vram_verdict = self._wrap(ttk.Label(self.vram_frame, text="",
                                                 style="Dim.TLabel",
                                                 justify="left"), 24)
        self.vram_verdict.pack(fill="x", pady=(self.px(2), 0))

        self.hint = ttk.Frame(self.vram_frame, style="Hint.TFrame")
        self.hint_lbl = tk.Label(self.hint, text="", bg=HINT_BG, fg=ACCENT,
                                 font=("Segoe UI", 9))
        self.hint_lbl.pack(side="left", padx=self.px(8), pady=self.px(4))
        use = self._link(self.hint, "Use →", app._use_max_ctx, bg=HINT_BG)
        use.pack(side="right", padx=self.px(8))

        self.vram_more = self._link(self.vram_frame, "", self._toggle_vram_details)
        self.vram_more.configure(fg=FG_DIM)
        self.vram_more.bind("<Button-1>", lambda e: self._toggle_vram_details())
        self.vram_more.pack(anchor="w", pady=(self.px(4), 0))
        self.vram_details = ttk.Frame(self.vram_frame)
        self.vram_cards = ttk.Frame(self.vram_details)
        self.vram_cards.pack(fill="x")
        self._link(self.vram_details, "Read usage now",
                   lambda: app._vram_follow_live(True)).pack(anchor="w")
        self.vram_notes = self._wrap(ttk.Label(self.vram_details, text="",
                                               style="Dim.TLabel",
                                               justify="left"), 24)
        self.vram_notes.pack(fill="x")
        self._vram_open = False
        self._toggle_vram_details(False)

    def _build_launch_bar(self, f):
        app = self.app
        bar = ttk.Frame(f, style="Bar.TFrame")
        bar.pack(side="bottom", fill="x")
        ttk.Separator(f).pack(side="bottom", fill="x")
        row = ttk.Frame(bar, style="Bar.TFrame")
        row.pack(fill="x", padx=self.px(12), pady=(self.px(8), 0))
        row.columnconfigure(0, weight=2, uniform="bar")
        row.columnconfigure(1, weight=1, uniform="bar")
        self.load_btn = ttk.Button(row, text="Launch", style="Go.TButton",
                                   command=app._load)
        self.load_btn.grid(row=0, column=0, sticky="ew", padx=(0, self.px(6)))
        ttk.Button(row, text="Preview", command=app._preview).grid(
            row=0, column=1, sticky="nsew")
        self.status_lbl = ttk.Label(bar, text="", style="Bar.TLabel",
                                    justify="left")
        self.status_lbl.pack(fill="x", padx=self.px(12), pady=(self.px(4), 0))
        self.replace_lbl = ttk.Label(bar, text="", style="Bar.TLabel",
                                     justify="left")
        self.replace_lbl.pack(fill="x", padx=self.px(12), pady=(0, self.px(8)))
        bar.bind("<Configure>", lambda e: [
            l.configure(wraplength=max(self.px(100), e.width - self.px(24)))
            for l in (self.status_lbl, self.replace_lbl)])
        app.one_at_a_time.trace_add("write", lambda *a: self._paint_replace())

    # ---- setup behaviour -------------------------------------------------

    def _pick_backend(self, name):
        if backends.get(name) is None or self.app.backend_var.get() == name:
            return
        self.app.backend_var.set(name)
        self.app._on_backend_change()

    def _paint_backends(self):
        chosen = self.app.backend_var.get()
        for name, (btn, status) in self.backend_btns.items():
            self._paint_toggle(btn, name == chosen)
            info = self._backend_status.get(name)
            if not info:
                status.configure(text="", style="Dim.TLabel")
                continue
            available, why, up = info
            if up:
                status.configure(text="running", style="Ok.TLabel")
            else:
                status.configure(text=why, style="Dim.TLabel" if available
                                 else "Bad.TLabel")

    def _pick_preset(self, name):
        self.app.preset_var.set(name)
        self.app._preset_chosen()

    def _paint_chips(self):
        chosen = self.app.preset_var.get()
        for name, chip in self._chip_btns.items():
            self._paint_toggle(chip, name == chosen)

    def _toggle_advanced(self, flip=True):
        if flip:
            self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self.adv_body.pack(fill="x", pady=(self.px(4), 0))
            self.adv_head.configure(text="▾ Advanced")
        else:
            self.adv_body.pack_forget()
            self.adv_head.configure(
                text="▸ Advanced · penalties, seed, keep alive, flags")

    def _toggle_vram_details(self, flip=True):
        if flip:
            self._vram_open = not self._vram_open
        if self._vram_open:
            self.vram_details.pack(fill="x", pady=(self.px(2), 0))
            self.vram_more.configure(text="▾ Details")
        else:
            self.vram_details.pack_forget()
            self.vram_more.configure(text="▸ Details · memory other "
                                     "apps use, per card")

    def _paint_replace(self):
        """Say what Launch will unload, when One model at a time is on."""
        ours = [i.label for i in self._loaded if not i.foreign_process]
        text = ""
        if ours and self.app.one_at_a_time.get():
            text = "Launch unloads %s first (one model at a time)." % ", ".join(ours)
        if self.replace_lbl.cget("text") != text:
            self.replace_lbl.configure(text=text)

    def _card(self, parent, caption, style="CardValue.TLabel"):
        card = ttk.Frame(parent, style="Card.TFrame", padding=self.px(8))
        cap = ttk.Label(card, text=caption, style="Card.TLabel")
        cap.pack(anchor="w")
        value = ttk.Label(card, text="-", style=style)
        value.pack(anchor="w")
        return card, cap, value

    def _build_monitor(self, f):
        """Three states, one shown at a time: loading (a launch is running),
        live (something is loaded), empty (nothing is)."""
        self.mon_loading = ttk.Frame(f)
        self.mon_empty = ttk.Frame(f)
        self.mon_live = ttk.Frame(f)
        self._build_loading(self.mon_loading)
        self._build_empty(self.mon_empty)
        self._build_live(self.mon_live)
        self._mon_panel = None
        self._show_mon_panel()

    def _build_empty(self, f):
        box = ttk.Frame(f)
        box.place(relx=0.5, rely=0.35, anchor="center")
        ttk.Label(box, text="No model running", style="Section.TLabel").pack()
        ttk.Label(box, text="Pick one in Setup and launch it.",
                  style="Dim.TLabel").pack(pady=(self.px(2), self.px(10)))
        ttk.Button(box, text="Go to Setup",
                   command=lambda: self.show("setup", user=True)).pack()

    def _build_loading(self, f):
        app = self.app
        pad = self.px(12)
        head = ttk.Frame(f)
        head.pack(fill="x", padx=pad, pady=(pad, 0))
        self.load_title = ttk.Label(head, text="", style="Section.TLabel")
        self.load_title.pack(side="left")
        self.load_elapsed = ttk.Label(head, text="", style="Dim.TLabel")
        self.load_elapsed.pack(side="right")
        self.load_bar = ttk.Progressbar(f, mode="indeterminate",
                                        style="Zoom.Thin.Horizontal.TProgressbar")
        self.load_bar.pack(fill="x", padx=pad, pady=(self.px(6), 0))
        self.load_step = ttk.Label(f, text="", style="Dim.TLabel")
        self.load_step.pack(fill="x", padx=pad, pady=(self.px(4), 0))

        buttons = ttk.Frame(f)
        buttons.pack(side="bottom", fill="x", padx=pad, pady=pad)
        for col in (0, 1):
            buttons.columnconfigure(col, weight=1, uniform="lb")
        self.cancel_btn = ttk.Button(buttons, text="Cancel",
                                     command=self._cancel_clicked)
        self.cancel_btn.grid(row=0, column=0, sticky="ew", padx=(0, self.px(4)))
        ttk.Button(buttons, text="Open log", command=app._open_log).grid(
            row=0, column=1, sticky="ew", padx=(self.px(4), 0))

        ttk.Label(f, text="Output", style="Section.TLabel").pack(
            anchor="w", padx=pad, pady=(self.px(10), self.px(2)))
        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True, padx=pad)
        # width=20: a Text asks for 80 characters by default, which would
        # make the whole window want to be that wide.
        self.load_out = tk.Text(wrap, bg=BG_PANEL, fg=FG_DIM, font=FONT_MONO,
                                relief="flat", wrap="char", height=8, width=20,
                                highlightthickness=1, highlightbackground=BORDER)
        self.load_out.pack(fill="both", expand=True)
        self.load_out.tag_configure("err", foreground=BAD)
        self.load_out.tag_configure("note", foreground=ACCENT)
        self.load_out.configure(state="disabled")

    def _build_live(self, f):
        app = self.app
        form = self._scrolled(f)
        pad = self.px(12)

        # which model: a chip each, and what the chosen one is
        top = ttk.Frame(form)
        top.pack(fill="x", padx=pad, pady=(pad, 0))
        self.model_chips = ttk.Frame(top)
        self.model_chips.pack(fill="x")
        self.model_info = self._wrap(ttk.Label(top, text="", style="Dim.TLabel",
                                               justify="left"), 24)
        self.model_info.pack(fill="x", pady=(self.px(4), 0))

        # speed
        box = ttk.Frame(form)
        box.pack(fill="x", padx=pad, pady=(self.px(10), 0))
        big, self.gen_caption, self.gen_value = self._card(
            box, "Generation", style="Big.TLabel")
        big.pack(fill="x")
        self.live_status = ttk.Label(big, text="", style="Card.TLabel")
        self.live_status.place(relx=1.0, x=-self.px(2), y=0, anchor="ne")
        pair = ttk.Frame(box)
        pair.pack(fill="x", pady=(self.px(6), 0))
        for col in (0, 1):
            pair.columnconfigure(col, weight=1, uniform="cards")
        card, _cap, self.ttft_value = self._card(pair, "TTFT")
        card.grid(row=0, column=0, sticky="ew", padx=(0, self.px(3)))
        card, _cap, self.prompt_value = self._card(pair, "Prompt eval")
        card.grid(row=0, column=1, sticky="ew", padx=(self.px(3), 0))
        head = ttk.Frame(box)
        head.pack(fill="x", pady=(self.px(8), 0))
        ttk.Label(head, text="Context", style="Dim.TLabel").pack(side="left")
        self.ctx_value = ttk.Label(head, text="-", style="Dim.TLabel")
        self.ctx_value.pack(side="right")
        self.ctx_bar = ttk.Progressbar(box, mode="determinate", maximum=1000,
                                       style="Zoom.Thin.Horizontal.TProgressbar")
        self.ctx_bar.pack(fill="x")
        ttk.Separator(form).pack(fill="x", pady=(self.px(12), 0))

        # every card, not just the first: this machine has two
        box, head = self._section(form, "GPUs")
        self.kv_value = ttk.Label(head, text="", style="Dim.TLabel")
        self.kv_value.pack(side="right")
        self.gpu_box = ttk.Frame(box)
        self.gpu_box.pack(fill="x")
        self.spill_lbl = self._wrap(ttk.Label(box, text="", style="Dim.TLabel",
                                              justify="left"), 24)
        self.spill_lbl.pack(fill="x", pady=(self.px(4), 0))

        # the request that finished last
        box, head = self._section(form, "Last request")
        self.last_when = ttk.Label(head, text="", style="Dim.TLabel")
        self.last_when.pack(side="right")
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        self.last_rows = {}
        for row, key in enumerate(("Prompt", "Generation", "First 3 s", "TTFT",
                                   "Total")):
            lab = ttk.Label(grid, text=key, style="Dim.TLabel")
            val = ttk.Label(grid, text="", justify="right", anchor="e")
            lab.grid(row=row, column=0, sticky="nw", pady=(0, self.px(2)))
            val.grid(row=row, column=1, sticky="ne", padx=(self.px(12), 0))
            self.last_rows[key] = (lab, val)
        self.last_none = ttk.Label(box, text="Nothing has been asked of it yet.",
                                   style="Off.TLabel")

        # actions
        buttons = ttk.Frame(form)
        buttons.pack(fill="x", padx=pad, pady=(self.px(4), pad))
        for col in (0, 1):
            buttons.columnconfigure(col, weight=1, uniform="mb")
        for i, (text, command) in enumerate((
                ("Unload", app._unload_selected), ("Unload all", app._unload_all),
                ("Open log", app._open_log), ("Copy endpoint", self._copy_endpoint))):
            row, col = divmod(i, 2)
            ttk.Button(buttons, text=text, command=command).grid(
                row=row, column=col, sticky="ew", pady=(0, self.px(6)),
                padx=(0, self.px(3)) if col == 0 else (self.px(3), 0))

    def _build_history(self, f):
        pad = self.px(12)
        top = ttk.Frame(f)
        top.pack(fill="x", padx=pad, pady=(self.px(8), 0))
        self.hist_chips = ttk.Frame(top)
        self.hist_chips.pack(fill="x")
        self.hist_summary = ttk.Label(top, text="", style="Dim.TLabel")
        self.hist_summary.pack(anchor="w", pady=(self.px(4), self.px(6)))

        bar = ttk.Frame(f)
        bar.pack(side="bottom", fill="x", padx=pad, pady=(self.px(6), pad))
        for col in (0, 1):
            bar.columnconfigure(col, weight=1, uniform="hb")
        ttk.Button(bar, text="Export CSV", command=self._export_history).grid(
            row=0, column=0, sticky="ew", padx=(0, self.px(4)))
        ttk.Button(bar, text="Clear", command=self._clear_history).grid(
            row=0, column=1, sticky="ew", padx=(self.px(4), 0))

        wrap = ttk.Frame(f)
        wrap.pack(fill="both", expand=True, padx=pad)
        scroll = ttk.Scrollbar(wrap, orient="vertical",
                               style="Dark.Vertical.TScrollbar")
        scroll.pack(side="right", fill="y")
        # The tree column carries time and model together: at this width one
        # wide column reads better than two narrow ones, and it leaves room
        # for a row's details to sit underneath it, indented.
        self.hist_tree = ttk.Treeview(wrap, columns=("gen", "ctx"),
                                      show="tree headings",
                                      style="Compact.Treeview",
                                      yscrollcommand=scroll.set)
        self.hist_tree.heading("#0", text="Time \u00b7 model", anchor="w")
        self.hist_tree.heading("gen", text="Gen", anchor="e")
        self.hist_tree.heading("ctx", text="Ctx", anchor="e")
        self.hist_tree.column("#0", width=self.px(150), minwidth=self.px(90),
                              stretch=True)
        self.hist_tree.column("gen", width=self.px(54), minwidth=self.px(40),
                              stretch=False, anchor="e")
        self.hist_tree.column("ctx", width=self.px(46), minwidth=self.px(36),
                              stretch=False, anchor="e")
        self.hist_tree.tag_configure("group", foreground=FG_DIM)
        self.hist_tree.tag_configure("detail", foreground=FG_DIM)
        self.hist_tree.pack(side="left", fill="both", expand=True)
        scroll.configure(command=self.hist_tree.yview)

    def _build_procs(self, f):
        pad = self.px(12)
        top = ttk.Frame(f)
        top.pack(fill="x", padx=pad, pady=(self.px(8), 0))
        line = ttk.Frame(top)
        line.pack(fill="x")
        self.procs_lbl = ttk.Label(line, text="", style="Small.TLabel")
        self.procs_lbl.pack(side="left")
        self._link(line, "Refresh", self.app._refresh_processes).pack(side="right")
        self.procs_left_lbl = ttk.Label(top, text="", style="SmallWarn.TLabel")
        self.procs_left_lbl.pack(anchor="w", pady=(self.px(2), 0))

        self.procs_box = self._scrolled(f)

    # ------------------------------------------------------------------
    # navigation
    # ------------------------------------------------------------------

    def show(self, key, user=False):
        if user:
            self._user_chose = True
        if key == self.current:
            return
        self.current = key
        self.frames[key].tkraise()
        self.title_lbl.configure(text=TITLES[key])
        self._paint_nav()
        if key == "procs":
            self.app._refresh_processes()

    def _paint_nav(self):
        for key, item in self.nav.items():
            active = key == self.current
            colour = ACCENT if active else FG_DIM
            text = item["text"]
            if key == "monitor" and not active:
                if self._launching:
                    text, colour = "Loading...", WARN
                elif self._generating_tps is not None:
                    text, colour = "%.0f t/s" % self._generating_tps, OK_GREEN
                elif self._loaded:
                    colour = OK_GREEN
            elif key == "procs" and self._leftovers and not active:
                text, colour = "Procs %d" % self._leftovers, WARN
            item["icon"].configure(fg=colour)
            item["label"].configure(fg=colour, text=text)

    def first_poll(self, loaded):
        """The first look at what is running. Something already loaded
        means there is nothing to set up - open on what it is doing."""
        if loaded and not self._user_chose:
            self.show("monitor")

    def launch_started(self):
        self._launching = True
        self._launch_t0 = time.time()
        plan = self.app.current_plan
        name = (plan.model_label or plan.model_id) if plan else ""
        self.load_title.configure(text="Loading %s" % name if name else "Loading")
        self.load_step.configure(text="Starting...")
        self.cancel_btn.configure(text="Cancel")
        self.cancel_btn.state(["!disabled"])
        self.load_out.configure(state="normal")
        self.load_out.delete("1.0", "end")
        self.load_out.configure(state="disabled")
        self.load_bar.start(15)
        self._tick_loading()
        self._show_mon_panel()
        self.show("monitor")
        self._paint_nav()

    def launch_finished(self, ok):
        self._launching = False
        if self._tick_id:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.load_bar.stop()
        self._show_mon_panel()
        if not ok:
            self.show("setup")    # the error is shown there, beside Launch
        self._paint_nav()

    def _tick_loading(self):
        self._tick_id = None
        if not self._launching:
            return
        secs = int(time.time() - self._launch_t0)
        self.load_elapsed.configure(text="%d:%02d" % (secs // 60, secs % 60)
                                    if secs >= 60 else "%d s" % secs)
        self._tick_id = self.root.after(1000, self._tick_loading)

    def close(self):
        """The window is going. A tick still pending would fire into a dead
        interpreter, which Tk complains about on the way out."""
        self._launching = False
        if self._tick_id:
            try:
                self.root.after_cancel(self._tick_id)
            except tk.TclError:
                pass
            self._tick_id = None

    def _cancel_clicked(self):
        self.cancel_btn.configure(text="Cancelling...")
        self.cancel_btn.state(["disabled"])
        self.app.cancel_launch()

    def _show_mon_panel(self):
        panel = (self.mon_loading if self._launching else
                 self.mon_live if self._loaded else self.mon_empty)
        if panel is self._mon_panel:
            return
        for other in (self.mon_loading, self.mon_live, self.mon_empty):
            other.pack_forget()
        panel.pack(fill="both", expand=True)
        self._mon_panel = panel

    def _copy_endpoint(self):
        chosen = self.selected_loaded()
        if chosen and chosen[0].endpoint:
            self.root.clipboard_clear()
            self.root.clipboard_append(chosen[0].endpoint)
            self.app.set_status("Copied %s" % chosen[0].endpoint, "Ok.TLabel")

    # ------------------------------------------------------------------
    # setup: what the controller calls
    # ------------------------------------------------------------------

    def show_models(self, labels):
        self.model_box.configure(values=labels)
        self.models_lbl.configure(text="%d model%s" % (
            len(labels), "" if len(labels) == 1 else "s") if labels else "")

    def show_presets(self, names):
        for chip in self._chip_btns.values():
            chip.destroy()
        for child in self.chips.winfo_children():
            child.destroy()
        self._chip_btns = {}
        if not names:
            ttk.Label(self.chips, text="None measured for this model",
                      style="Off.TLabel").pack(anchor="w")
            return
        for name in names:
            chip = self._toggle(self.chips, name, lambda n=name: self._pick_preset(n))
            chip.pack(side="left", padx=(0, self.px(6)))
            self._chip_btns[name] = chip
        self._paint_chips()

    def show_modes(self, labels):
        self.mode_box.configure(values=labels)
        self.mode_box.state(["!disabled"] if labels else ["disabled"])

    def show_reasoning(self, levels, enabled):
        self.reason_box.configure(values=levels)
        self.reason_box.state(["!disabled"] if enabled else ["disabled"])

    def show_kv(self, choices, enabled):
        self.kv_box.configure(values=choices)
        self.kv_box.state(["!disabled"] if enabled else ["disabled"])
        self.kv_label.configure(style="Dim.TLabel" if enabled else "Off.TLabel")

    def set_field_supported(self, key, supported):
        self.entries[key].state(["!disabled"] if supported else ["disabled"])
        self.labels[key].configure(style="Dim.TLabel" if supported
                                   else "Off.TLabel")

    def set_field_auto(self, key, auto):
        """Blue for what the docs or a preset filled in, white for what you
        typed - the same rule as the classic layout."""
        try:
            self.entries[key].configure(foreground=ACCENT if auto else FG)
        except tk.TclError:
            pass

    def set_folder_state(self, editable, can_download):
        self._folder_editable, self._can_download = editable, can_download

    def set_apply_enabled(self, enabled):
        self.apply_link.configure(fg=ACCENT if enabled else FG_DIM,
                                  cursor="hand2" if enabled else "")

    def set_launch_enabled(self, enabled):
        self.load_btn.state(["!disabled"] if enabled else ["disabled"])

    def set_status(self, text, style="Dim.TLabel"):
        self.status_lbl.configure(text=text, foreground=TONE.get(style, FG))

    def set_source(self, text):
        self.source_lbl.configure(text=text)

    def set_notes(self, text):
        self.notes_lbl.configure(text=text)

    def log(self, text, tag=None):
        self.log_lines.append((text, tag))
        if not self._launching:
            return
        # While loading, the Output box follows along, last 200 lines.
        self.load_out.configure(state="normal")
        self.load_out.insert("end", text + "\n", tag or ())
        extra = int(self.load_out.index("end-1c").split(".")[0]) - 200
        if extra > 0:
            self.load_out.delete("1.0", "%d.0" % (extra + 1))
        self.load_out.see("end")
        self.load_out.configure(state="disabled")
        if text.startswith("[zoomies] "):
            self.load_step.configure(text=text[len("[zoomies] "):])

    # ---- VRAM estimate ---------------------------------------------------

    def show_vram_cards(self, cards, variables, on_slide):
        """A slider per card for what other apps hold on it; they follow
        the measured value until moved."""
        for child in self.vram_cards.winfo_children():
            child.destroy()
        self._vram_labels = {}
        for card in cards:
            head = ttk.Frame(self.vram_cards)
            head.pack(fill="x", pady=(self.px(4), 0))
            ttk.Label(head, text="%s - other apps" % card.name,
                      style="Dim.TLabel").pack(side="left")
            lbl = ttk.Label(head, text="", style="Dim.TLabel")
            lbl.pack(side="right")
            ttk.Scale(self.vram_cards, from_=0, to=card.total / float(vram.GB),
                      variable=variables[card.luid],
                      style="Dark.Horizontal.TScale",
                      command=lambda v, l=card.luid: on_slide(l)).pack(fill="x")
            self._vram_labels[card.luid] = lbl

    def set_vram_card_text(self, luid, text):
        lbl = self._vram_labels.get(luid)
        if lbl is not None:
            lbl.configure(text=text)

    def show_vram_estimate(self, est, text, style, notes, verdict=None):
        """One bar per card - this machine has two - coloured by how close
        that card comes to full, then the verdict and the context hint."""
        self._est = est
        for child in self.vram_bars.winfo_children():
            child.destroy()
        if est is None:
            self.vram_verdict.configure(text=text, style=style)
            self.vram_notes.configure(text="")
            self.hint.pack_forget()
            self.vram_more.pack_forget()
            self.vram_details.pack_forget()
            return
        if not self.vram_more.winfo_manager():
            self.vram_more.pack(anchor="w", pady=(self.px(4), 0))
            self._toggle_vram_details(False)
        for card in est.cards:
            tone = "Bad" if card.spare < 0 else (
                "Warn" if card.spare < est.margin else "Ok")
            head = ttk.Frame(self.vram_bars)
            head.pack(fill="x", pady=(self.px(4), 0))
            ttk.Label(head, text=card.name, style="Dim.TLabel").pack(side="left")
            ttk.Label(head, text="%s / %.0f GB" % (
                vram.gb(card.used), card.total / float(vram.GB)),
                style="Dim.TLabel").pack(side="right")
            bar = ttk.Progressbar(self.vram_bars, mode="determinate",
                                  maximum=max(1, card.total),
                                  style="%s.Thin.Horizontal.TProgressbar" % tone)
            bar.configure(value=min(card.used, card.total))
            bar.pack(fill="x")
        # The split behind each bar goes under Details: too long for a row
        # this narrow, and only wanted when the total looks wrong.
        split = ["%s: %s GB other apps + %s GB this model" % (
            c.name, vram.gb(c.other), vram.gb(c.model)) for c in est.cards]
        self.vram_notes.configure(text="\n".join(split + [notes]))
        verdict = verdict or text
        self.vram_verdict.configure(text=verdict[:1].upper() + verdict[1:],
                                    style=style)
        if est.max_ctx and est.max_ctx != est.ctx:
            self.hint_lbl.configure(
                text=("Room for %s context" if est.max_ctx > est.ctx
                      else "Fits up to %s context") % fmt_k(est.max_ctx))
            self.hint.pack(fill="x", pady=(self.px(6), 0), before=self.vram_more)
        else:
            self.hint.pack_forget()

    # ------------------------------------------------------------------
    # monitoring: what the controller calls
    # ------------------------------------------------------------------

    def show_backend_status(self, status):
        if status != self._backend_status:
            self._backend_status = dict(status)
            self._paint_backends()

    @staticmethod
    def _key(item):
        return (item.backend, item.endpoint, item.label)

    @staticmethod
    def _set(lbl, text, **kw):
        """Configure only on change: this runs five times a second."""
        if lbl.cget("text") != text or kw:
            lbl.configure(text=text, **kw)

    def show_loaded(self, loaded):
        changed = bool(loaded) != bool(self._loaded)
        # Keyed like the classic table, so the same model twice is one chip.
        seen, items = set(), []
        for item in loaded:
            if self._key(item) not in seen:
                seen.add(self._key(item))
                items.append(item)
        self._loaded = items
        keys = tuple(self._key(i) for i in items)
        if keys != self._chips_key:
            self._chips_key = keys
            if self._picked not in keys:
                self._picked = keys[0] if keys else None
            for child in self.model_chips.winfo_children():
                child.destroy()
            for item in items:
                chip = self._toggle(self.model_chips, item.label,
                                    lambda k=self._key(item): self._pick_model(k))
                chip.pack(side="left", padx=(0, self.px(6)), pady=(0, self.px(4)))
                chip.key = self._key(item)
            self._paint_model_chips()
        self._show_model_info()
        self._show_mon_panel()
        if changed:
            self._paint_nav()
        self._paint_replace()

    def _pick_model(self, key):
        self._picked = key
        self._paint_model_chips()
        self._show_model_info()

    def _paint_model_chips(self):
        for chip in self.model_chips.winfo_children():
            self._paint_toggle(chip, getattr(chip, "key", None) == self._picked)

    def _show_model_info(self):
        chosen = self.selected_loaded()
        if not chosen:
            self._set(self.model_info, "")
            return
        item = chosen[0]
        be = backends.get(item.backend)
        port = item.endpoint.rsplit(":", 1)[-1] if item.endpoint else ""
        self._set(self.model_info, join(
            be.display_name if be else item.backend,
            ":" + port if port else "",
            "pid %s" % item.pid if item.pid else "",
            human_bytes(item.vram_bytes) if item.vram_bytes else "",
            "%s ctx" % fmt_k(item.context) if item.context else "",
            "until %s" % until(item.expires) if item.expires else "",
            "" if item.owned_by_us else "not started by Zoomies"))

    def selected_loaded(self):
        return [i for i in self._loaded if self._key(i) == self._picked][:1]

    def show_live(self, snap):
        status = snap.get("status") or ""
        tps = snap.get("tg") if status == "Generating..." else None
        if tps != self._generating_tps:
            self._generating_tps = tps
            self._paint_nav()
        history = snap.get("history") or []
        self._show_speed(snap, status, history)
        self._show_gpus(snap)
        self._show_last(history[0] if history else None)
        self._show_history(history)

    def _show_speed(self, snap, status, history):
        model = snap.get("model") or ""
        if status == "Generating...":
            value, caption = snap.get("tg"), "Generation \u00b7 now"
        else:
            value = snap.get("gen_avg")
            if value is None and history:
                value = history[0].get("gen_avg")
            caption = "Generation \u00b7 last request"
        self._set(self.gen_value, fmt_tps(value) or "-")
        self._set(self.gen_caption, join(caption, model))
        if status == "Processing prompt..." and snap.get("prompt_progress"):
            status = "Prompt %d%%" % round(100 * snap["prompt_progress"])
        self._set(self.live_status, status.rstrip("."),
                  foreground=OK_GREEN if status == "Generating..." else FG_DIM)
        self._set(self.ttft_value, metrics.fmt_ttft(snap.get("ttft"),
                                                    snap.get("ttft_upper")))
        self._set(self.prompt_value, fmt_tps(snap.get("prompt_tps")) or "-")
        used, total = snap.get("n_tokens"), snap.get("n_ctx")
        if used and total:
            self.ctx_bar.configure(value=min(1000, int(1000.0 * used / total)))
            self._set(self.ctx_value, "%s / %s" % (fmt_k(used), fmt_k(total)))
        else:
            self.ctx_bar.configure(value=0)
            self._set(self.ctx_value, "-")
        self._set(self.kv_value, "KV %s" % snap["kv"] if snap.get("kv") else "")

    def _show_gpus(self, snap):
        gpus = snap.get("gpus") or []
        if len(gpus) != len(self._gpu_rows):
            for child in self.gpu_box.winfo_children():
                child.destroy()
            self._gpu_rows = []
            for _gpu in gpus:
                head = ttk.Frame(self.gpu_box)
                head.pack(fill="x", pady=(self.px(4), 0))
                name = ttk.Label(head, text="", style="Dim.TLabel")
                name.pack(side="left")
                mem = ttk.Label(head, text="", style="Dim.TLabel")
                mem.pack(side="right")
                bar = ttk.Progressbar(self.gpu_box, mode="determinate",
                                      maximum=1000,
                                      style="Zoom.Thin.Horizontal.TProgressbar")
                bar.pack(fill="x")
                self._gpu_rows.append((name, mem, bar))
        for gpu, (name, mem, bar) in zip(gpus, self._gpu_rows):
            pct, used = gpu.get("pct"), gpu.get("used")
            self._set(name, join(gpu["name"], "" if pct is None
                                 else "%.0f%% busy" % pct))
            if used is None:
                self._set(mem, "-")
                bar.configure(value=0)
            else:
                self._set(mem, "%s / %.0f GB" % (vram.gb(used),
                                                 gpu["vram"] / float(vram.GB)))
                bar.configure(value=min(1000, int(1000.0 * used
                                                  / max(1, gpu["vram"]))))
        spills = snap.get("spills") or []
        if spills:
            text = "Spilling into system RAM: " + ", ".join(
                "%.1f GB (pid %d)" % (sp["shared"] / float(vram.GB), sp["pid"])
                for sp in spills)
            self._set(self.spill_lbl, text, style="Bad.TLabel")
        elif snap.get("gpu_error"):
            self._set(self.spill_lbl, "GPU: %s" % snap["gpu_error"],
                      style="Bad.TLabel")
        elif gpus:
            self._set(self.spill_lbl, "No spill into system RAM",
                      style="Off.TLabel")

    def _show_last(self, row):
        if row is None:
            if not self.last_none.winfo_manager():
                self.last_none.pack(anchor="w")
            self._set(self.last_when, "")
            for lab, val in self.last_rows.values():
                lab.grid_remove()
                val.grid_remove()
            return
        self.last_none.pack_forget()
        self._set(self.last_when, join(row.get("time"), row.get("model")))
        cached = row.get("prompt_cached")
        prompt = join(
            "%s tok" % metrics.fmt_int(row["prompt_n"]) if row.get("prompt_n") else "",
            "+%s cached" % metrics.fmt_int(cached) if cached else "",
            fmt_secs(row.get("prompt_s")),
            fmt_tps(row.get("prompt_tps")))
        gen = join(
            "%s tok" % metrics.fmt_int(row["n_gen"]) if row.get("n_gen") else "",
            fmt_secs(row.get("gen_s")),
            fmt_tps(row.get("gen_avg")))
        values = {
            "Prompt": prompt,
            "Generation": gen,
            "First 3 s": fmt_tps(row.get("tg3s")),
            "TTFT": metrics.fmt_ttft(row.get("ttft"), row.get("ttft_upper"))
            if row.get("ttft") is not None else "",
            "Total": fmt_secs(row.get("runtime")) or "",
        }
        for key, (lab, val) in self.last_rows.items():
            if values[key]:
                self._set(val, values[key])
                lab.grid()
                val.grid()
            else:
                lab.grid_remove()
                val.grid_remove()

    # ---- history ---------------------------------------------------------

    def _show_history(self, rows):
        """Rebuild only when the rows or the filter actually changed: this
        is called five times a second."""
        key = (len(rows), rows[0].get("time") if rows else "",
               rows[0].get("model") if rows else "", self._hist_filter)
        if key == self._hist_key:
            return
        self._hist_key = key
        self._paint_hist_chips(rows)
        kept = [r for r in rows if self._hist_filter in (None, r.get("model"))]
        self._hist_rows = kept
        speeds = [r["gen_avg"] for r in kept if r.get("gen_avg")]
        self.hist_summary.configure(text=join(
            "%d of %d kept" % (len(kept), len(rows)) if len(kept) != len(rows)
            else "%d request%s" % (len(rows), "" if len(rows) == 1 else "s"),
            "avg %.1f t/s" % (sum(speeds) / len(speeds)) if speeds else ""))

        self.hist_tree.delete(*self.hist_tree.get_children(""))
        if not kept:
            self.hist_tree.insert("", "end", text="Nothing recorded yet",
                                  tags=("group",))
            return
        day = None
        for i, row in enumerate(kept):
            if row.get("date") != day:
                day = row.get("date")
                self.hist_tree.insert("", "end", text=day_label(day),
                                      tags=("group",))
            node = self.hist_tree.insert(
                "", "end", iid="r%d" % i, tags=("row",),
                text=join(row.get("time", "")[:5], short_model(row)),
                values=(metrics.fmt(row.get("gen_avg")),
                        fmt_k(row.get("n_tokens"))))
            for text in detail_lines(row):
                self.hist_tree.insert(node, "end", text=text, tags=("detail",))

    def _paint_hist_chips(self, rows):
        names, seen = [], set()
        for row in rows:                     # newest first, so is the order
            name = row.get("model")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        wanted = ["All"] + names[:5]
        if [c.cget("text") for c in self.hist_chips.winfo_children()] != wanted:
            for child in self.hist_chips.winfo_children():
                child.destroy()
            for name in wanted:
                chip = self._toggle(self.hist_chips, name,
                                    lambda n=name: self._filter_history(n))
                chip.pack(side="left", padx=(0, self.px(4)), pady=(0, self.px(4)))
        for chip in self.hist_chips.winfo_children():
            self._paint_toggle(chip, chip.cget("text")
                               == (self._hist_filter or "All"))

    def _filter_history(self, name):
        self._hist_filter = None if name == "All" else name
        self._hist_key = None                # force a rebuild
        self._show_history(self._all_history())

    def _all_history(self):
        return (self.app.metrics.snapshot().get("history") or []) \
            if self.app.metrics else []

    def _export_history(self):
        rows = self._hist_rows
        if not rows:
            self.app.set_status("Nothing to export.", "Warn.TLabel")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export history",
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="zoomies-history-%s.csv"
            % datetime.date.today().isoformat())
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except OSError as exc:
            self.app.set_status("Could not write it: %s" % exc, "Bad.TLabel")
            return
        self.app.set_status("Exported %d request%s." % (
            len(rows), "" if len(rows) == 1 else "s"), "Ok.TLabel")

    def _clear_history(self):
        if not self.app.metrics:
            return
        if not messagebox.askyesno(
                "Clear history?",
                "Forget all %d recorded request%s? The models and their "
                "settings are not touched." % (
                    len(self._all_history()),
                    "" if len(self._all_history()) == 1 else "s"),
                parent=self.root):
            return
        self.app.metrics.clear_history()
        self._hist_key = None
        self._show_history([])

    def show_procs(self, items):
        """items: the processes.Proc list, or None when it could not be read."""
        if items is None:
            self._set(self.procs_lbl, "Could not read the process list.",
                      style="SmallWarn.TLabel")
            return
        self.procs = items
        left = [p for p in items if p.leftover]
        self._set(self.procs_lbl, "%d running \u00b7 %s" % (
            len(items), processes.fmt_ram(sum(p.ram for p in items))),
            style="Small.TLabel")
        self._set(self.procs_left_lbl, "%d leftover%s \u00b7 %s" % (
            len(left), "" if len(left) == 1 else "s",
            processes.fmt_ram(sum(p.ram for p in left))) if left else "")
        if len(left) != self._leftovers:
            self._leftovers = len(left)
            self._paint_nav()

        key = tuple((p.pid, p.what, p.ram // (64 * 1024 ** 2)) for p in items)
        if key == self._procs_key:
            return                        # same list, give or take 64 MB
        self._procs_key = key
        for child in self.procs_box.winfo_children():
            child.destroy()
        if not items:
            ttk.Label(self.procs_box, text="Nothing of ours is running.",
                      style="SmallOff.TLabel").pack(anchor="w",
                                                    padx=self.px(12))
            return
        for head, group in (("Leftovers \u00b7 nothing accounts for these", left),
                            ("Accounted for", [p for p in items
                                               if not p.leftover])):
            if not group:
                continue
            ttk.Label(self.procs_box, text=head, style="SmallDim.TLabel").pack(
                anchor="w", padx=self.px(12), pady=(self.px(10), self.px(2)))
            for proc in group:
                self._proc_row(proc)
            if group is left:
                ttk.Button(self.procs_box, style="Small.TButton",
                           text="Clean up %d leftover%s" % (
                               len(left), "" if len(left) == 1 else "s"),
                           command=self.app._clean_leftovers).pack(
                    fill="x", padx=self.px(12), pady=(self.px(6), 0))

    def _proc_row(self, proc):
        row = ttk.Frame(self.procs_box)
        row.pack(fill="x", padx=self.px(12), pady=(self.px(2), 0))
        row.columnconfigure(0, weight=1)
        what = ttk.Label(row, text=proc.what, justify="left",
                         style="SmallWarn.TLabel" if proc.leftover
                         else "Small.TLabel")
        what.grid(row=0, column=0, sticky="w")
        self._wrap(what, 90)
        if proc.protected:
            ttk.Label(row, text="kept", style="SmallOff.TLabel").grid(
                row=0, column=1, sticky="e")
        elif proc.zoomies_port or "Ollama unloads it" in proc.what:
            # Unloading is the right way to stop these; ending them by hand
            # would leave the backend believing they are still there.
            ttk.Label(row, text="unload", style="SmallOff.TLabel").grid(
                row=0, column=1, sticky="e")
        else:
            ttk.Button(row, text="End", style="Small.TButton",
                       command=lambda p=proc: self._end_proc(p)).grid(
                row=0, column=1, sticky="e", padx=(self.px(6), 0))
        started = proc.started[11:16] if len(proc.started) > 15 else proc.started
        ttk.Label(row, text=join("pid %d" % proc.pid,
                                 processes.fmt_ram(proc.ram), started,
                                 proc.parent or "parent exited"),
                  style="SmallDim.TLabel").grid(row=1, column=0, columnspan=2,
                                                sticky="w")

    def _end_proc(self, proc):
        """The controller asks before ending anything, and says what it will
        end - so the row is offered to it as the selection."""
        self._proc_pick = proc.pid
        self.app._end_selected()

    def selected_procs(self):
        return [p for p in self.procs if p.pid == self._proc_pick][:1]
