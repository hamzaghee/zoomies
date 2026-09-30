"""Keep the DeepSeek Harness (dsh) in step with the running llama.cpp server.

The same problem opencode.py solves, for a different client and a different
file format. dsh reads the context window out of its own settings.yaml and
never asks the server: `llm-deepseek.models[].contextWindow`, falling back to
`defaultContextWindow`, is what decides when it compacts. A number left over
from an earlier load therefore decides it wrongly, and dsh's own default is
1,000,000 - far past anything a local server serves.

Three things go out of step on a load, so three things are written:

  * `contextWindow` for the model that is up - the context it is serving is
    decided at load time, and llama.cpp can hand out less than was asked for.
  * `maxTokens` - dsh does not clamp the reply budget against the window
    (its own default is 256,000), so the window alone is not enough.
  * `agent-default-model.model` - llama.cpp serves one model at a time, so a
    selection naming anything else cannot be served at all.

settings.yaml is edited by line rather than reparsed and dumped: the comments
in it explain why each number is what it is, and a YAML round-trip would
throw them away. Only the values move; the layout around them is untouched.
"""

import hashlib
import os
import re

import backends

# Same shape as opencode.py's: a local endpoint we can pull a port out of.
_LOCAL_URL = re.compile(r"^https?://(?:127\.0\.0\.1|localhost):(\d+)")

# The settings section this module owns. Nothing outside it is touched,
# except the one model selection in `agent-default-model`.
_SECTION = "llm-deepseek"

# Indents in dsh's settings.yaml: a top-level section's own keys sit at two
# spaces, a model entry's keys at six. Matching the exact indent is what
# keeps a section-level `maxTokens` from being confused with a per-model one.
_KEY_INDENT = 2
_ENTRY_INDENT = 6


def config_path():
    """dsh's settings document: $DSH_HOME/settings.yaml, else ~/.dsh."""
    home = os.environ.get("DSH_HOME")
    if not home:
        home = os.path.join(os.path.expanduser("~"), ".dsh")
    return os.path.join(home, "settings.yaml")


def config_stamp(path=None):
    """A digest of the settings file, or None if it is not there.

    Contents rather than modified time, for the reason opencode.py gives:
    65536 and 131072 are the same six bytes to a timestamp that moves in
    16 ms steps.
    """
    try:
        with open(path or config_path(), "rb") as f:
            return hashlib.blake2b(f.read(), digest_size=16).digest()
    except OSError:
        return None


def _yaml_single(value):
    """A YAML single-quoted scalar: backslashes literal, quotes doubled.

    The model ids here are Windows paths, because Zoomies starts llama.cpp
    with `--alias <full gguf path>` and that alias is the id the server
    advertises. Single quotes are the only style that carries a backslash
    through without escaping it.
    """
    return "'" + str(value).replace("'", "''") + "'"


def _unquote(raw):
    """The value of a YAML scalar as written on one line."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        inner = raw[1:-1]
        return inner.replace("''", "'") if raw[0] == "'" else inner
    return raw


class Plan:
    """The edits for one sync, as (start, end, replacement) on the text."""

    def __init__(self, path, text):
        self.path, self.text = path, text
        self.edits = []
        self.changed = []
        self.skipped = []
        self.notes = []

    @property
    def new_text(self):
        out = self.text
        for start, end, repl in sorted(self.edits, key=lambda e: e[0],
                                       reverse=True):
            out = out[:start] + repl + out[end:]
        return out

    def summary(self):
        parts = []
        if self.changed:
            parts.append("Will change:\n"
                         + "\n".join("  " + c for c in self.changed))
        else:
            parts.append("Nothing to change - dsh already matches.")
        if self.skipped:
            parts.append("Left alone:\n"
                         + "\n".join("  " + s for s in self.skipped))
        if self.notes:
            parts.append("Notes:\n" + "\n".join("  " + n for n in self.notes))
        return "\n\n".join(parts)


def _section_span(text, name):
    """(start, end) of a top-level block, ending at the next top-level key."""
    head = re.search(r"^%s:[ \t]*$" % re.escape(name), text, re.M)
    if not head:
        return None
    nxt = re.search(r"^[^\s#][^\n]*$", text[head.end():], re.M)
    if not nxt:
        return head.end(), len(text)
    return head.end(), head.end() + nxt.start()


def _scalar_edit(plan, text, span, key, value, label, indent):
    """Replace `key: <value>` at `indent` inside `span`.

    True when an edit was queued, False when the value already matches, and
    None when the key is not there at all - which the caller has to decide
    about, because an absent key means dsh falls back to its own default
    rather than to anything we wrote.
    """
    pat = re.compile(r"^( {%d}%s:[ \t]*)(.*?)[ \t]*$" % (indent, re.escape(key)),
                     re.M)
    for m in pat.finditer(text, span[0], span[1]):
        if m.group(2) == str(value):
            return False
        plan.edits.append((m.start(2), m.end(2), str(value)))
        plan.changed.append("%s: %s -> %s"
                            % (label, m.group(2) or "unset", value))
        return True
    return None


def _model_entries(text, span):
    """[(id, entry_start, entry_end, id_line_end)] for each `- id:` entry."""
    models = re.search(r"^[ \t]+models:[ \t]*$", text[span[0]:span[1]], re.M)
    if not models:
        return []
    start = span[0] + models.end()
    heads = list(re.finditer(r"^[ \t]+-[ \t]+id:[ \t]*(.+?)[ \t]*$",
                             text[start:span[1]], re.M))
    out = []
    for i, m in enumerate(heads):
        end = start + heads[i + 1].start() if i + 1 < len(heads) else span[1]
        out.append((_unquote(m.group(1)), start + m.start(), end,
                    start + m.end()))
    return out


def _live_by_port(loaded=None):
    """{(port, id): model} for every llama.cpp server that is up."""
    if loaded is None:
        loaded = backends.get("llamacpp").list_loaded()
    live = {}
    for model in loaded:
        port = _LOCAL_URL.match(str(model.endpoint or ""))
        if model.backend != "llamacpp" or not model.context or not port:
            continue
        for key in (model.id, model.label):
            if key:
                live[(int(port.group(1)), key)] = model
    return live


def plan_limits(path=None, loaded=None):
    """Edits that put the running server's real context into dsh's settings.

    Only a model that is up is touched. An entry for something not loaded
    keeps whatever it has: it says nothing about a server that isn't there.
    """
    path = path or config_path()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    plan = Plan(path, text)

    span = _section_span(text, _SECTION)
    if span is None:
        plan.skipped.append("%s: no such section in %s"
                            % (_SECTION, os.path.basename(path)))
        return plan
    live = _live_by_port(loaded)
    if not live:
        return plan

    base = re.search(r"^[ \t]+baseURL:[ \t]*(\S+)[ \t]*$",
                     text[span[0]:span[1]], re.M)
    port_match = _LOCAL_URL.match(_unquote(base.group(1))) if base else None
    if not port_match:
        plan.skipped.append("%s: baseURL is not a local llama.cpp endpoint"
                            % _SECTION)
        return plan
    port = int(port_match.group(1))

    entries = _model_entries(text, span)
    model = next((live[(port, mid)] for mid, *_ in entries
                  if (port, mid) in live), None)

    if model is None:
        # A server is up on this port serving something the catalog does not
        # list - a freshly downloaded gguf, or an alias that changed. Said
        # rather than written: which name to file it under, and what else to
        # say about it, is a choice the full sync makes with the user
        # watching, not one to make behind a poll.
        here = [m for (p, _), m in live.items() if p == port]
        if here:
            plan.notes.append(
                "port %d is serving %r, which %s.models does not list - add it "
                "there to pick it in dsh" % (port, here[0].id, _SECTION))
        return plan

    # 1. the loaded model's own contextWindow
    for mid, entry_start, entry_end, id_line_end in entries:
        if mid != model.id:
            continue
        got = _scalar_edit(plan, text, (entry_start, entry_end),
                           "contextWindow", model.context, "contextWindow",
                           _ENTRY_INDENT)
        if got is None:
            plan.edits.append((id_line_end, id_line_end, "\n%scontextWindow: %d"
                               % (" " * _ENTRY_INDENT, model.context)))
            plan.changed.append("contextWindow: unset -> %d" % model.context)
        break

    # 2. the reply budget, which dsh does not derive from the window
    want_max = backends.max_tokens_for(model.context)
    if want_max and _scalar_edit(plan, text, span, "maxTokens", want_max,
                                 "maxTokens", _KEY_INDENT) is None:
        plan.notes.append("%s.maxTokens is not set; dsh will fall back to its "
                          "own 256000 default" % _SECTION)

    # 3. the selection, which cannot name a model the server is not serving
    adm = _section_span(text, "agent-default-model")
    if adm:
        _scalar_edit(plan, text, adm, "model", _yaml_single(model.id),
                     "selected model", _KEY_INDENT)
    return plan


def write(plan, backup=True):
    """Commit the plan atomically. Returns the backup path, or None."""
    if not plan.edits:
        return None
    saved = None
    if backup:
        saved = plan.path + ".bak"
        with open(saved, "w", encoding="utf-8", newline="") as f:
            f.write(plan.text)
    tmp = plan.path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(plan.new_text)
    os.replace(tmp, plan.path)
    return saved
