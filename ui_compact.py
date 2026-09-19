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

from ui_common import (ACCENT, BG, BG_FIELD, FG, FG_DIM, OK_GREEN, WARN)

# Segoe Fluent Icons ships with Windows 11 and MDL2 Assets with Windows 10;
# both put the same glyphs at the same code points.
ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")
GLYPH = {
    "setup": "",        # Equalizer
    "monitor": "",      # Diagnostic
    "history": "",      # History
    "procs": "",        # Processing
    "more": "",         # More
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
        self._build()
        self.show("setup")

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _glyph(self, key):
        return GLYPH[key] if self.icon_font else FALLBACK[key]

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

    def _build_setup(self, f):
        self._placeholder(f, "The new setup form is still being built. To "
                          "change settings or launch a model meanwhile, pick "
                          "Switch to classic layout in the menu.")
        ttk.Label(f, text="Model", style="Dim.TLabel").pack(
            fill="x", padx=self.px(12), pady=(self.px(6), 0))
        self._line(f, self.app.model_var)
        self.status_lbl = self._line(f, style="Dim.TLabel")

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
        pass

    def show_presets(self, names):
        pass

    def show_modes(self, labels):
        pass

    def show_reasoning(self, levels, enabled):
        pass

    def show_kv(self, choices, enabled):
        pass

    def set_field_supported(self, key, supported):
        pass

    def set_field_auto(self, key, auto):
        pass

    def set_folder_state(self, editable, can_download):
        self._folder_editable, self._can_download = editable, can_download

    def set_apply_enabled(self, enabled):
        pass

    def set_launch_enabled(self, enabled):
        pass

    def set_status(self, text, style="Dim.TLabel"):
        self.status_lbl.configure(text=text, style=style)

    def set_source(self, text):
        pass

    def set_notes(self, text):
        pass

    def log(self, text, tag=None):
        self.log_lines.append((text, tag))

    def show_vram_cards(self, cards, variables, on_slide):
        pass

    def set_vram_card_text(self, luid, text):
        pass

    def show_vram_estimate(self, est, text, style, notes):
        pass

    # ------------------------------------------------------------------
    # monitoring: what the controller calls
    # ------------------------------------------------------------------

    def show_backend_status(self, status):
        pass

    def show_loaded(self, loaded):
        changed = bool(loaded) != bool(self._loaded)
        self._loaded = list(loaded)
        text = "\n".join("%s  -  %s" % (i.label, i.endpoint.replace("http://", ""))
                         for i in loaded) or "Nothing loaded."
        if self.loaded_lbl.cget("text") != text:
            self.loaded_lbl.configure(text=text)
        if changed:
            self._paint_nav()

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
