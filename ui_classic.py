r"""
Zoomies - the classic layout: one wide window, everything stacked.

A layout only draws. Every value it shows lives on the controller (app.py)
as a tk variable, and every button calls a controller method, so the
same logic runs whichever layout is on screen. What the controller needs
back from a layout is the set of show_*/set_* methods below; the compact
layout implements the same ones.
"""

import tkinter as tk
from tkinter import ttk

import backends
import metrics
import processes
import vram
from ui_common import (ACCENT, BAD, BG_PANEL, BORDER, FG, FG_DIM, FONT_MONO,
                       WARN, human_bytes, until)


class ClassicLayout:
    name = "classic"
    size = (980, 800)             # at 100% display scale
    min_size = (780, 560)
    remember_geometry = False     # opens centred, as it always has
    dock = None

    def __init__(self, app):
        self.app = app
        self.root = app.root
        self.px = app.px
        self.entries, self.labels = {}, {}
        self.loaded_rows = {}             # Loaded table: row id -> LoadedModel
        self.procs = []
        self._vram_labels = {}            # luid -> value label beside its slider
        self._build()

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _build(self):
        app = self.app
        pad = {"padx": 8, "pady": 3}

        # ---- top bar -------------------------------------------------
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(top, text="Zoomies", style="Head.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Checkbutton(top, text="Always on top", variable=app.on_top,
                        command=app._toggle_top).pack(side="right")
        ttk.Checkbutton(top, text="Unload on exit",
                        variable=app.unload_exit).pack(side="right", padx=(0, 12))
        ttk.Button(top, text="Sync opencode...",
                   command=app._sync_opencode).pack(side="right", padx=(0, 12))
        for name in app.other_layouts():
            ttk.Button(top, text="Try %s layout" % name,
                       command=lambda n=name: app.switch_layout(n)).pack(
                side="right", padx=(0, 12))

        # ---- backend + model ----------------------------------------
        pick = ttk.Frame(self.root)
        pick.pack(fill="x", **pad)
        pick.columnconfigure(1, weight=1)

        ttk.Label(pick, text="Backend").grid(row=0, column=0, sticky="w")
        row = ttk.Frame(pick)
        row.grid(row=0, column=1, sticky="ew")
        self.backend_status = {}
        for name in ("ollama", "llamacpp"):
            be = backends.get(name)
            rb = ttk.Radiobutton(
                row, text=(be.display_name if be else name.title()),
                value=name, variable=app.backend_var,
                command=app._on_backend_change)
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
        self.folder_entry = ttk.Entry(frow, textvariable=app.folder_var)
        self.folder_entry.grid(row=0, column=0, sticky="ew")
        self.browse_btn = ttk.Button(frow, text="Browse...", command=app._browse)
        self.browse_btn.grid(row=0, column=1, padx=(6, 0))
        ttk.Button(frow, text="Open", command=app._open_folder).grid(
            row=0, column=2, padx=(6, 0))
        ttk.Button(frow, text="Rescan", command=app._reload_models).grid(
            row=0, column=3, padx=(6, 0))
        self.download_btn = ttk.Button(frow, text="Download...",
                                       command=app._download)
        self.download_btn.grid(row=0, column=4, padx=(6, 0))

        ttk.Label(pick, text="Model").grid(row=2, column=0, sticky="w", pady=3)
        self.model_box = ttk.Combobox(pick, textvariable=app.model_var,
                                      state="readonly")
        self.model_box.grid(row=2, column=1, sticky="ew", pady=3)
        self.model_box.bind("<<ComboboxSelected>>",
                            lambda e: app.model_picked(self.model_box.current()))

        # ---- settings ------------------------------------------------
        box = ttk.LabelFrame(self.root, text=" Settings ")
        box.pack(fill="x", padx=8, pady=(8, 3))

        bar = ttk.Frame(box)
        bar.pack(fill="x", padx=8, pady=(6, 2))
        self.apply_btn = ttk.Button(bar, text="Fill from docs",
                                    command=app._apply_optimal)
        self.apply_btn.pack(side="left")
        ttk.Label(bar, text="Mode").pack(side="left", padx=(14, 4))
        self.mode_box = ttk.Combobox(bar, textvariable=app.mode_var,
                                     state="readonly", width=22)
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", lambda e: app._mode_changed())
        ttk.Label(bar, text="Reasoning").pack(side="left", padx=(14, 4))
        self.reason_box = ttk.Combobox(bar, textvariable=app.reason_var,
                                       state="readonly", width=14)
        self.reason_box.pack(side="left")
        self.reason_box.bind("<<ComboboxSelected>>",
                             lambda e: app._reason_changed())
        ttk.Label(bar, text="Preset").pack(side="left", padx=(14, 4))
        self.preset_box = ttk.Combobox(bar, textvariable=app.preset_var,
                                       state="readonly", width=18)
        self.preset_box.pack(side="left")
        self.preset_box.bind("<<ComboboxSelected>>",
                             lambda e: app._preset_chosen())
        ttk.Button(bar, text="Clear", command=app._clear_settings).pack(
            side="left", padx=(10, 0))
        # Answers are saved permanently once found, so there has to be a way
        # to go back and look again when the docs change.
        ttk.Button(bar, text="Re-read docs",
                   command=lambda: app._apply_optimal(force=True)).pack(
            side="left", padx=(6, 0))

        grid = ttk.Frame(box)
        grid.pack(fill="x", padx=8, pady=4)

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
            # Small minimum width: the columns stretch to fill the window
            # anyway, and the default 20 characters made the grid wider
            # than the window at its minimum size.
            ent = ttk.Entry(grid, textvariable=app.vars[key], justify="left",
                            width=8)
            ent.grid(row=row, column=col * 2 + 1, sticky="ew",
                     padx=(0, self.px(18)), pady=3,
                     columnspan=(span * 2 - 1) if span > 1 else 1)
            self.entries[key], self.labels[key] = ent, lab

        row = 0
        for row, keys in enumerate(backends.SETTING_ROWS):
            for col, key in enumerate(keys):
                if key:
                    make_field(key, row, col)
        # KV cache takes the free cell beside Parallel: it is a load-time
        # setting like its neighbours, and the top bar has no room left.
        kv_row = len(backends.SETTING_ROWS) - 1
        self.kv_label = ttk.Label(grid, text="KV cache", style="Dim.TLabel",
                                  anchor="e")
        self.kv_label.grid(row=kv_row, column=6, sticky="e", padx=(0, 6), pady=3)
        self.kv_box = ttk.Combobox(grid, textvariable=app.kv_var,
                                   state="readonly", width=8,
                                   values=backends.KV_CACHE_CHOICES)
        self.kv_box.grid(row=kv_row, column=7, sticky="ew",
                         padx=(0, self.px(18)), pady=3)
        self.kv_box.bind("<<ComboboxSelected>>", lambda e: app._kv_changed())

        for key in backends.SETTING_WIDE:
            row += 1
            make_field(key, row, 0, span=4)
        # --chat-template-file changes which template sets the levels.
        self.entries["extra_flags"].bind(
            "<FocusOut>", lambda e: app._refresh_reasoning())

        self.source_lbl = ttk.Label(box, text="", style="Dim.TLabel",
                                    wraplength=self.px(940), justify="left")
        self.source_lbl.pack(fill="x", padx=8, pady=(2, 0))
        self.notes_lbl = ttk.Label(box, text="", style="Warn.TLabel",
                                   wraplength=self.px(940), justify="left")
        self.notes_lbl.pack(fill="x", padx=8, pady=(2, 6))
        self._build_vram(box)

        act = ttk.Frame(box)
        act.pack(fill="x", padx=8, pady=(0, 8))
        self.load_btn = ttk.Button(act, text="Load model", style="Go.TButton",
                                   command=app._load)
        self.load_btn.pack(side="left")
        ttk.Button(act, text="Preview script",
                   command=app._preview).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(act, text="One model at a time",
                        variable=app.one_at_a_time).pack(side="left", padx=(12, 0))
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
        self.tree.bind("<Double-1>", lambda e: app._unload_selected())

        lbar = ttk.Frame(lbox)
        lbar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(lbar, text="Unload selected",
                   command=app._unload_selected).pack(side="left")
        ttk.Button(lbar, text="Unload all",
                   command=app._unload_all).pack(side="left", padx=(8, 0))
        ttk.Button(lbar, text="Temporary tags...",
                   command=app._manage_tags).pack(side="left", padx=(8, 0))
        self.vram_lbl = ttk.Label(lbar, text="", style="Dim.TLabel")
        self.vram_lbl.pack(side="right")

        self._build_live(self.root)

    def _build_live(self, parent):
        """Compact live-metrics strip plus a tabbed History/Output pane.

        Tabs rather than three stacked panes: the window is already tall, and
        History and Output are rarely both wanted at once.
        """
        app = self.app
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
        for key, label in (("prompt", "Prompt eval (avg)"), ("ttft", "TTFT"),
                           ("gen", "Generation"), ("tokens", "Generated"),
                           ("kv", "KV cache")):
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
        cols = ("time", "backend", "model", "ttft", "prompt", "gen",
                "runtime", "tokens", "kv")
        widths = (65, 75, 185, 60, 95, 90, 65, 245, 70)
        headings = {"time": "Time", "backend": "Backend", "model": "Model",
                    "ttft": "TTFT", "prompt": "Prompt t/s (avg)",
                    "gen": "Gen t/s (avg)", "runtime": "Runtime",
                    "tokens": "Tokens (context used)", "kv": "KV"}
        self.hist_tree = ttk.Treeview(hist, columns=cols, show="headings",
                                      height=7)
        for col, w in zip(cols, widths):
            self.hist_tree.heading(col, text=headings[col])
            self.hist_tree.column(col, width=self.px(w), minwidth=self.px(40),
                                  anchor="w" if col in ("model", "tokens")
                                  else "center")
        self.hist_tree.pack(fill="both", expand=True)

        out = ttk.Frame(tabs)
        tabs.add(out, text="  Output  ")
        bar = ttk.Frame(out)
        bar.pack(fill="x", pady=(4, 2))
        ttk.Button(bar, text="Open log", command=app._open_log).pack(side="right")
        ttk.Button(bar, text="Open folder",
                   command=app._open_scripts).pack(side="right", padx=(0, 6))
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

        # Every llama.cpp / Ollama process, including ones no API reports -
        # a terminal chat, a server whose launcher exited - so a slow machine
        # can be explained and cleaned up without Task Manager.
        pw = ttk.Frame(tabs)
        tabs.add(pw, text="  Processes  ")
        self.procs_tab = pw
        pbar = ttk.Frame(pw)
        pbar.pack(fill="x", pady=(4, 2))
        ttk.Button(pbar, text="Refresh",
                   command=app._refresh_processes).pack(side="left")
        ttk.Button(pbar, text="Clean up leftovers",
                   command=app._clean_leftovers).pack(side="left", padx=(8, 0))
        ttk.Button(pbar, text="End selected",
                   command=app._end_selected).pack(side="left", padx=(8, 0))
        self.procs_lbl = ttk.Label(pbar, text="", style="Dim.TLabel")
        self.procs_lbl.pack(side="left", padx=(16, 0))
        cols = ("what", "ram", "pid", "started", "parent")
        widths = (560, 75, 65, 95, 120)
        headings = {"what": "What it is", "ram": "RAM", "pid": "PID",
                    "started": "Started", "parent": "Started by"}
        self.proc_tree = ttk.Treeview(pw, columns=cols, show="headings", height=7)
        for col, w in zip(cols, widths):
            self.proc_tree.heading(col, text=headings[col])
            self.proc_tree.column(col, width=self.px(w), minwidth=self.px(40),
                                  anchor="w" if col in ("what", "parent") else "center")
        self.proc_tree.pack(fill="both", expand=True)
        self.proc_tree.tag_configure("leftover", foreground=WARN)
        self.proc_tree.tag_configure("protected", foreground=FG_DIM)
        tabs.bind("<<NotebookTabChanged>>",
                  lambda e: app._refresh_processes()
                  if tabs.select() == str(pw) else None)
        self.tabs = tabs

    def _build_vram(self, box):
        """What the load will need per card, before loading it.

        Worked out from the .gguf header (vram.py), so it never touches a
        GPU. The sliders are what everything else already holds on each
        card; they follow Windows' own counters until you move one.
        """
        app = self.app
        frame = ttk.Frame(box)
        frame.pack(fill="x", padx=8, pady=(0, 6))
        head = ttk.Frame(frame)
        head.pack(fill="x")
        ttk.Label(head, text="VRAM", style="Dim.TLabel").pack(side="left")
        self.vram_est_lbl = ttk.Label(head, text="", style="Dim.TLabel",
                                      wraplength=self.px(640), justify="left")
        self.vram_est_lbl.pack(side="left", padx=(8, 0))
        self.vram_fit_btn = ttk.Button(head, text="Use largest context",
                                       command=app._use_max_ctx)
        self.vram_fit_btn.pack(side="right")
        ttk.Button(head, text="Read usage now",
                   command=lambda: app._vram_follow_live(True)).pack(
            side="right", padx=(0, 6))
        self.vram_cards = ttk.Frame(frame)
        self.vram_cards.pack(fill="x", pady=(2, 0))
        self.vram_notes = ttk.Label(frame, text="", style="Dim.TLabel",
                                    wraplength=self.px(940), justify="left")
        self.vram_notes.pack(fill="x")

    # ------------------------------------------------------------------
    # moments the compact layout moves between views on; this one shows
    # everything at once, so it has nothing to do
    # ------------------------------------------------------------------

    def first_poll(self, loaded):
        pass

    def launch_started(self):
        pass

    def launch_finished(self, ok):
        pass

    # ------------------------------------------------------------------
    # setup: what the controller calls
    # ------------------------------------------------------------------

    def show_models(self, labels):
        self.model_box.configure(values=labels)

    def show_presets(self, names):
        self.preset_box.configure(values=names)
        self.preset_box.state(["!disabled"] if names else ["disabled"])

    def show_modes(self, labels):
        self.mode_box.configure(values=labels)

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
        """Auto-filled values render in accent blue so it is obvious at a
        glance which numbers came from the docs and which you typed."""
        try:
            self.entries[key].configure(foreground=ACCENT if auto else FG)
        except tk.TclError:
            pass

    def set_folder_state(self, editable, can_download):
        self.folder_entry.state(["!disabled"] if editable else ["disabled"])
        self.browse_btn.state(["!disabled"] if editable else ["disabled"])
        self.download_btn.state(["!disabled"] if can_download else ["disabled"])

    def set_apply_enabled(self, enabled):
        self.apply_btn.state(["!disabled"] if enabled else ["disabled"])

    def set_launch_enabled(self, enabled):
        self.load_btn.state(["!disabled"] if enabled else ["disabled"])

    def set_status(self, text, style="Dim.TLabel"):
        self.status_lbl.configure(text=text, style=style)

    def set_source(self, text):
        self.source_lbl.configure(text=text)

    def set_notes(self, text):
        self.notes_lbl.configure(text=text)

    def log(self, text, tag=None):
        self.out.configure(state="normal")
        self.out.insert("end", text + "\n", tag or ())
        self.out.see("end")
        self.out.configure(state="disabled")

    # ---- VRAM estimate ---------------------------------------------------

    def show_vram_cards(self, cards, variables, on_slide):
        """One slider per card the launch will use. variables: luid ->
        DoubleVar of GB in use by other apps, owned by the controller."""
        for child in self.vram_cards.winfo_children():
            child.destroy()
        self._vram_labels = {}
        for row, card in enumerate(cards):
            gb = card.total / float(vram.GB)
            ttk.Label(self.vram_cards, text="%s - in use by other apps"
                      % card.name, style="Dim.TLabel").grid(
                row=row, column=0, sticky="w")
            ttk.Scale(self.vram_cards, from_=0, to=gb,
                      variable=variables[card.luid], length=self.px(220),
                      command=lambda v, l=card.luid: on_slide(l)).grid(
                row=row, column=1, padx=8)
            lbl = ttk.Label(self.vram_cards, text="", style="Dim.TLabel")
            lbl.grid(row=row, column=2, sticky="w")
            self._vram_labels[card.luid] = lbl

    def set_vram_card_text(self, luid, text):
        lbl = self._vram_labels.get(luid)
        if lbl is not None:
            lbl.configure(text=text)

    def show_vram_estimate(self, est, text, style, notes, verdict=None):
        """est: the vram.Estimate behind text, or None when there is none.
        verdict: just the fits / tight / spills part of text."""
        self.vram_est_lbl.configure(text=text, style=style)
        self.vram_notes.configure(text=notes)
        if est is not None and est.max_ctx and est.max_ctx != est.ctx:
            self.vram_fit_btn.configure(text="Use largest context (%s)"
                                        % format(est.max_ctx, ","))
            self.vram_fit_btn.state(["!disabled"])
        else:
            self.vram_fit_btn.configure(text="Use largest context")
            self.vram_fit_btn.state(["disabled"])

    # ------------------------------------------------------------------
    # monitoring: what the controller calls
    # ------------------------------------------------------------------

    def show_backend_status(self, status):
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

    def show_loaded(self, loaded):
        # Rows are keyed by which model on which endpoint, not by position.
        # Numbering them meant that when one model replaced another, the row
        # that was on screen kept its place and simply changed its text - so a
        # model that had just been unloaded looked like the new one arriving
        # under the wrong name. Now the old row goes and a new one appears.
        existing = set(self.tree.get_children(""))
        wanted = []
        self.loaded_rows = {}
        for item in loaded:
            iid = "%s|%s|%s" % (item.backend, item.endpoint, item.label)
            if iid in self.loaded_rows:                # same model twice over
                continue
            wanted.append(iid)
            self.loaded_rows[iid] = item
            values = (
                item.label,
                backends.get(item.backend).display_name if backends.get(item.backend)
                else item.backend,
                human_bytes(item.vram_bytes),
                format(item.context, ",") if item.context else "-",
                item.endpoint.replace("http://", "") or "-",
                item.pid or "-",
                until(item.expires),
            )
            tag = "ours" if item.owned_by_us else "foreign"
            if iid in existing:
                self.tree.item(iid, values=values, tags=(tag,))
            else:
                self.tree.insert("", "end", iid=iid, values=values, tags=(tag,))
        for iid in existing - set(wanted):
            self.tree.delete(iid)
        for position, iid in enumerate(wanted):
            self.tree.move(iid, "", position)

        total = sum(i.vram_bytes for i in loaded)
        self.vram_lbl.configure(
            text="VRAM in use: %s" % human_bytes(total) if total else "")

    def selected_loaded(self):
        return [self.loaded_rows[r] for r in self.tree.selection()
                if r in self.loaded_rows]

    def show_live(self, snap):
        backend = snap.get("backend") or ""
        label = backend and (backends.get(backend).display_name
                             if backends.get(backend) else backend)
        # The model too, not just the backend: with servers coming and going
        # under a benchmark run, "which model is this" is the first thing
        # anyone looking at these numbers wants to know.
        if label and snap.get("model"):
            label = "%s   -   %s" % (label, snap["model"])
        status = snap.get("status") or ""
        self.live_status.configure(
            text="%s%s" % (status, "   -   %s" % label if label else ""),
            style="Ok.TLabel" if status == "Generating..." else "Head.TLabel")

        self.live_stats["prompt"].configure(
            text=metrics.fmt(snap.get("prompt_tps"), " t/s"))
        self.live_stats["ttft"].configure(
            text=metrics.fmt_ttft(snap.get("ttft"), snap.get("ttft_upper")))
        self.live_stats["gen"].configure(text=metrics.fmt(snap.get("tg"), " t/s"))
        self.live_stats["tokens"].configure(
            text=metrics.fmt_int(snap.get("n_gen")))
        self.live_stats["kv"].configure(text=snap.get("kv") or "-")

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
                pct, used = gpu.get("pct"), gpu.get("used")
                parts.append("%s %s%s" % (
                    gpu["name"], "-" if pct is None else "%.0f%%" % pct,
                    "" if used is None else "  %s/%.0f GB" % (
                        vram.gb(used), gpu["vram"] / float(vram.GB))))
            spilling = bool(snap.get("spills"))
            if spilling:
                parts.append("SPILLING INTO SYSTEM RAM")
            self.live_gpu.configure(text="   ".join(parts),
                                    style="Bad.TLabel" if spilling else "Dim.TLabel")
        elif snap.get("gpu_error"):
            self.live_gpu.configure(text="GPU: %s" % snap["gpu_error"],
                                    style="Bad.TLabel")

        self._show_history(snap.get("history") or [])

    def _show_history(self, rows):
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
                metrics.fmt_ttft(row.get("ttft"), row.get("ttft_upper")),
                metrics.fmt(row.get("prompt_tps")),
                metrics.fmt(row.get("gen_avg")),
                metrics.fmt_mmss(row.get("runtime")),
                self._token_cell(row),
                row.get("kv") or "-",
            )
            if iid in existing:
                self.hist_tree.item(iid, values=values)
            else:
                self.hist_tree.insert("", "end", iid=iid, values=values)
        for iid in existing - wanted:
            self.hist_tree.delete(iid)

    @staticmethod
    def _token_cell(row):
        """Context used as a bar, with the tokens this request generated."""
        cell = metrics.bar_text(row.get("n_tokens"), row.get("n_ctx"))
        if cell != "-" and row.get("n_gen"):
            cell += "  (+%s)" % metrics.fmt_int(row.get("n_gen"))
        return cell

    # ---- processes -------------------------------------------------------

    def show_procs(self, items):
        """items: the processes.Proc list, or None when it could not be read."""
        if items is None:
            self.procs_lbl.configure(text="Could not read the process list.",
                                     style="Bad.TLabel")
            return
        self.procs = items
        self.proc_tree.delete(*self.proc_tree.get_children(""))
        for i, p in enumerate(items):
            started = p.started[5:16].replace("T", " ") if p.started else "-"
            tag = "leftover" if p.leftover else ("protected" if p.protected else "")
            self.proc_tree.insert("", "end", iid=str(i), tags=(tag,) if tag else (),
                                  values=(("LEFTOVER   " if p.leftover else "") + p.what,
                                          processes.fmt_ram(p.ram), p.pid, started,
                                          p.parent or "(exited)"))
        left = [p for p in items if p.leftover]
        text = "%d processes using %s" % (len(items),
                                          processes.fmt_ram(sum(p.ram for p in items)))
        if left:
            text += "  -  %d leftover%s using %s" % (
                len(left), "" if len(left) == 1 else "s",
                processes.fmt_ram(sum(p.ram for p in left)))
        self.procs_lbl.configure(text=text,
                                 style="Warn.TLabel" if left else "Dim.TLabel")
        self.tabs.tab(self.procs_tab, text=(
            "  Processes (%d leftover%s)  " % (len(left), "" if len(left) == 1 else "s")
            if left else "  Processes  "))

    def selected_procs(self):
        return [self.procs[int(r)] for r in self.proc_tree.selection()
                if 0 <= int(r) < len(self.procs)]
