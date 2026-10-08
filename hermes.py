"""Keep Hermes in step with the running llama.cpp server.

The same problem opencode.py and dsh.py solve, for a third client. Hermes
differs from both in one way that shapes this whole module: it does not hold a
catalogue of models. It holds ONE selection - `model.default`, plus the
`fallback_providers` entries that back it up - because llama.cpp serves one
model at a time, so a fallback naming a different model would 404 rather than
help. There is nothing here to add a model to; there is only a pin to keep
current, and before this module nothing did that, so it went stale on every
swap.

Three things go out of step on a load, so three things are written:

  * `model.context_length` - Hermes pins this because with `default` blank it
    has no model id to probe and assumes 256,000 tokens, which badly overruns
    a local -c and breaks compression on long tool loops.
  * `model.default` - the id Hermes sends as the model name. Zoomies starts
    llama.cpp with `--alias <full gguf path>`, so that path verbatim is the id
    the server advertises, and anything else 404s.
  * `fallback_providers[].model` - an entry is silently DISCARDED unless both
    `provider` and `model` are non-empty (agent_init._fallback_entries), so the
    id has to be spelled out there too even though `default` is set.

config.yaml is edited by line rather than reparsed and dumped, for the reason
dsh.py gives: the comments in it explain why each number is what it is, and a
YAML round-trip would throw them away. Only the values move.

Three deliberate safety choices, because this file is dense with comments and
is not ours:

  * Nothing is ever inserted. A key that is not there is reported, not
    created - dsh.py inserts a missing `contextWindow` because it owns that
    entry, but here an absent key means Hermes falls back to its own default
    and guessing where to put a new line in someone else's commented file is
    not worth the risk.
  * Every edit is scoped to a section span. `model:` at four spaces also
    appears under the voice and reference_models sections ("model: base",
    "model: whisper-1", "model: anthropic/..."), and a loose pattern would
    happily rewrite those.
  * The result is parsed before it is committed, when PyYAML is importable.
    A write that would leave the file unloadable raises instead.
"""

import hashlib
import os
import re

import backends

# Same shape as dsh.py's: a local endpoint we can pull a port out of.
_LOCAL_URL = re.compile(r"^https?://(?:127\.0\.0\.1|localhost):(\d+)")

# Indents in Hermes's config.yaml: a top-level section's own keys sit at two
# spaces, and a fallback_providers entry's keys at four.
_KEY_INDENT = 2
_ENTRY_INDENT = 4


def config_path():
    """Hermes's config document.

    The real file is under %LOCALAPPDATA%\\hermes; the .env in that folder
    points at ~/.hermes/config.yaml, which is where it used to live, so both
    are tried and whichever exists wins.
    """
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    first = os.path.join(local, "hermes", "config.yaml")
    second = os.path.join(os.path.expanduser("~"), ".hermes", "config.yaml")
    return first if os.path.isfile(first) or not os.path.isfile(second) \
        else second


def config_stamp(path=None):
    """A digest of the config file, or None if it is not there.

    Contents rather than modified time, for the reason opencode.py gives:
    65536 and 131072 are the same six bytes to a timestamp that moves in
    16 ms steps.
    """
    try:
        with open(path or config_path(), "rb") as f:
            return hashlib.blake2b(f.read(), digest_size=16).digest()
    except OSError:
        return None


# These four helpers are deliberately a copy of dsh.py's rather than an import
# of its private names: the two clients' files can then never break each other,
# and a change made for one is not silently inherited by the other.

def _yaml_single(value):
    """A YAML single-quoted scalar: backslashes literal, quotes doubled."""
    return "'" + str(value).replace("'", "''") + "'"


def _unquote(raw):
    """The value of a YAML scalar as written on one line."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        inner = raw[1:-1]
        return inner.replace("''", "'") if raw[0] == "'" else inner
    return raw


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
    None when the key is not there at all.
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
            parts.append("Nothing to change - Hermes already matches.")
        if self.skipped:
            parts.append("Left alone:\n"
                         + "\n".join("  " + s for s in self.skipped))
        if self.notes:
            parts.append("Notes:\n" + "\n".join("  " + n for n in self.notes))
        return "\n\n".join(parts)


def _local_provider(text):
    """(provider name, port) for the first provider pointing at a local
    llama.cpp endpoint, or (None, None).

    Hermes names its providers freely - the one here is "llama-8080" - so the
    name is read rather than assumed, and it is what the fallback entries are
    matched on below.
    """
    span = _section_span(text, "providers")
    if span is None:
        return None, None
    body = text[span[0]:span[1]]
    name = None
    for line in body.splitlines():
        head = re.match(r"^ {2}([^\s#:][^:]*):[ \t]*$", line)
        if head:
            name = head.group(1).strip()
            continue
        url = re.match(r"^ {4}base_url:[ \t]*(\S+)[ \t]*$", line)
        if url and name:
            port = _LOCAL_URL.match(_unquote(url.group(1)))
            if port:
                return name, int(port.group(1))
    return None, None


def _fallback_entries(text):
    """[(provider, entry_start, entry_end)] for each fallback_providers entry."""
    span = _section_span(text, "fallback_providers")
    if span is None:
        return []
    heads = list(re.finditer(r"^ {2}-[ \t]+provider:[ \t]*(.+?)[ \t]*$",
                             text[span[0]:span[1]], re.M))
    out = []
    for i, m in enumerate(heads):
        end = span[0] + heads[i + 1].start() if i + 1 < len(heads) else span[1]
        out.append((_unquote(m.group(1)), span[0] + m.start(), end))
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
    """Edits that put the running server's model and context into Hermes.

    Nothing is written when no local server is up: Hermes keeping the last
    model pinned is better than Hermes pinned to nothing.
    """
    path = path or config_path()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    plan = Plan(path, text)

    name, port = _local_provider(text)
    if not name:
        plan.skipped.append(
            "%s: no provider with a local llama.cpp base_url"
            % os.path.basename(path))
        return plan

    live = _live_by_port(loaded)
    if not live:
        return plan
    model = next((m for (p, _), m in sorted(live.items()) if p == port), None)
    if model is None:
        plan.notes.append("nothing is loaded on port %d, so the pin is left "
                          "as it is" % port)
        return plan

    # 1. the context Hermes sizes its packing against
    span = _section_span(text, "model")
    if span is None:
        plan.skipped.append("no top-level `model:` section to pin")
    else:
        if _scalar_edit(plan, text, span, "context_length", model.context,
                        "context_length", _KEY_INDENT) is None:
            plan.notes.append("model.context_length is not set; Hermes will "
                              "assume 256000 and overrun this server's -c")
        if _scalar_edit(plan, text, span, "default", _yaml_single(model.id),
                        "default model", _KEY_INDENT) is None:
            plan.notes.append("model.default is not set; Hermes will probe for "
                              "an id and can send one llama-server 404s")

    # 2. the fallback entries for this provider, which are discarded if their
    #    model is empty and 404 if it names anything else
    for provider, start, end in _fallback_entries(text):
        if provider != name:
            continue
        if _scalar_edit(plan, text, (start, end), "model",
                        _yaml_single(model.id), "fallback model",
                        _ENTRY_INDENT) is None:
            plan.notes.append("a fallback_providers entry for %r has no "
                              "`model:` line, so Hermes discards it" % provider)
    return plan


def write(plan, backup=True):
    """Commit the plan atomically. Returns the backup path, or None.

    The new text is parsed before it replaces the old one when PyYAML is
    importable, so a bad edit fails loudly here instead of quietly leaving
    Hermes with a config it cannot load. PyYAML missing is not an error; it
    only means the check is skipped.
    """
    if not plan.edits:
        return None
    new_text = plan.new_text
    try:
        import yaml
    except ImportError:
        pass
    else:
        try:
            yaml.safe_load(new_text)
        except yaml.YAMLError as exc:
            raise ValueError("refusing to write %s: the result would not parse "
                             "as YAML (%s)" % (os.path.basename(plan.path), exc))
    saved = None
    if backup:
        saved = plan.path + ".bak"
        with open(saved, "w", encoding="utf-8", newline="") as f:
            f.write(plan.text)
    tmp = plan.path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    os.replace(tmp, plan.path)
    return saved
