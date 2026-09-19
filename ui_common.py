r"""
Zoomies - what every layout shares: the palette, fonts, the ttk style sheet
and the little formatters the panels use to turn numbers into text.

Kept apart from app.py so a layout can import it without importing the
controller that imports the layout.
"""

import datetime
from tkinter import ttk

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


def human_bytes(n):
    if not n:
        return "-"
    return "%.1f GB" % (n / 1024.0 ** 3)


def until(expires):
    """How long until an Ollama keep-alive runs out, as 12m or 1.5h."""
    if not expires:
        return "-"
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


def apply_style(root, px):
    """The dark ttk theme. px scales a 100%-display size to this display."""
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
                 rowheight=px(22))
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
                 darkcolor=ACCENT, thickness=px(12))
    st.configure("TNotebook", background=BG, bordercolor=BORDER,
                 tabmargins=(2, 4, 2, 0))
    st.configure("TNotebook.Tab", background=BG_FIELD,
                 foreground=FG_DIM, padding=(14, 5))
    st.map("TNotebook.Tab",
           background=[("selected", BG_PANEL)],
           foreground=[("selected", ACCENT)])

    root.option_add("*TCombobox*Listbox.background", BG_FIELD)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", "#094771")
