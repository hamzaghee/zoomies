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
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import backends
import vram
from ui_common import (ACCENT, BAD, BG, BG_FIELD, BG_PANEL, BORDER, FG, FG_DIM,
                       OK_GREEN, WARN)

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


def short_ctx(n):
    """65536 -> 64k, 50000 -> 49k: what the context hint has room for."""
    return "%dk" % round(n / 1024.0) if n >= 1024 else str(n)


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
        self._backend_status = {}
        self._est = None
        self._vram_labels = {}
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
        st.configure("Bar.TFrame", background=BG_PANEL)
        st.configure("Bar.TLabel", background=BG_PANEL, foreground=FG_DIM)
        st.configure("Hint.TFrame", background=HINT_BG)
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
            for lbl, spare in self._wrapping:
                lbl.configure(wraplength=max(self.px(100), event.width - spare))
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
        """Wrap lbl to the form's width, less spare pixels of padding."""
        self._wrapping.append((lbl, self.px(spare)))
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
        ttk.Label(box, text="Preset", style="Dim.TLabel").pack(anchor="w")
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

    def _build_monitor(self, f):
        self._placeholder(f, "The monitor is still being built. What is "
                          "running right now:")
        self.live_lbl = self._line(f, style="Head.TLabel")
        self.loaded_lbl = self._line(f)

    def _build_history(self, f):
        self._placeholder(f, "History is still being built.")
        self.history_lbl = self._line(f)

    def _build_procs(self, f):
        self._placeholder(f, "The process list is still being built. The "
                          "classic layout has it under Processes.")
        self.procs_lbl = self._line(f)

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
        self.show("monitor")
        self._paint_nav()

    def launch_finished(self, ok):
        self._launching = False
        if not ok:
            self.show("setup")    # the error is shown there, beside Launch
        self._paint_nav()

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
                      else "Fits up to %s context") % short_ctx(est.max_ctx))
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

    def show_loaded(self, loaded):
        changed = bool(loaded) != bool(self._loaded)
        self._loaded = list(loaded)
        text = "\n".join("%s  -  %s" % (i.label, i.endpoint.replace("http://", ""))
                         for i in loaded) or "Nothing loaded."
        if self.loaded_lbl.cget("text") != text:
            self.loaded_lbl.configure(text=text)
        if changed:
            self._paint_nav()
        self._paint_replace()

    def selected_loaded(self):
        return self._loaded[:1]

    def show_live(self, snap):
        status = snap.get("status") or ""
        model = snap.get("model") or ""
        text = "%s%s" % (status, "  -  %s" % model if model else "")
        if self.live_lbl.cget("text") != text:
            self.live_lbl.configure(text=text)
        tps = snap.get("tg") if status == "Generating..." else None
        if tps != self._generating_tps:
            self._generating_tps = tps
            self._paint_nav()
        count = len(snap.get("history") or [])
        text = "%d recent request%s." % (count, "" if count == 1 else "s")
        if self.history_lbl.cget("text") != text:
            self.history_lbl.configure(text=text)

    def show_procs(self, items):
        if items is None:
            self.procs_lbl.configure(text="Could not read the process list.")
            return
        left = [p for p in items if p.leftover]
        self.procs_lbl.configure(text="%d running, %d leftover%s."
                                 % (len(items), len(left),
                                    "" if len(left) == 1 else "s"))
        if len(left) != self._leftovers:
            self._leftovers = len(left)
            self._paint_nav()

    def selected_procs(self):
        return []
