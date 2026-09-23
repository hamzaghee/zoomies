r"""
Zoomies - keep opencode's reasoning dropdown in step with the models.

opencode builds a model's reasoning dropdown from two places: levels it
guesses from the model's name, and the "variants" in opencode.jsonc. Its
guesses do not fit llama.cpp models. Any name containing "qwen" gets none, so
Qwen3.8's effort levels never appear, and everything else gets low / medium /
high sent as reasoning_effort, which the Ornith, Gemma and Nemotron templates
never read.

So this writes, for each model the config already lists on a Zoomies port:

    "reasoning": true/false        whether the model can reason at all
    "variants": {                  one per level its chat template accepts,
      "off": {"chat_template_kwargs": {"enable_thinking": false}},  ...
      "medium": {"disabled": true} opencode's guesses that do nothing
    }
    "options": {                   the sampling numbers from the model's
      "temperature": 0.15, ...     preset - and only those
    }

The options block matters because opencode sends it with every request,
and a request's own value beats whatever the server was started with: a
number left there quietly overrules Zoomies, which is the one place these
are meant to be set. So where a preset has a number it is written here,
and where it has none the setting is removed and the server's own value
stands. Settings Zoomies does not own - a timeout, a header - are left
alone, as are names, limits and comments. An agent's "variant" is only
touched when it names a variant this replaced, and then only if a new
level sends the same kwargs.

Levels are listed least reasoning first on purpose: opencode runs titles and
summaries with a model's first variant.
"""

import json
import os
import re
import time

import backends
import reasoning
import state

# opencode's own guesses (transform.ts, variants()): no levels for these
# names, low / medium / high for any other reasoning model on
# @ai-sdk/openai-compatible.
OPENCODE_NO_GUESS = ("deepseek-chat", "deepseek-reasoner", "deepseek-r1",
                     "deepseek-v3", "minimax", "glm", "kimi", "k2p", "qwen",
                     "big-pickle")
OPENCODE_GUESSES = ("low", "medium", "high")

_LOCAL_URL = re.compile(r"^https?://(?:127\.0\.0\.1|localhost):(\d+)")


def config_path():
    override = os.environ.get("OPENCODE_CONFIG")
    if override:
        return override
    folder = os.path.join(os.path.expanduser("~"), ".config", "opencode")
    for name in ("opencode.jsonc", "opencode.json"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return os.path.join(folder, "opencode.jsonc")


# --------------------------------------------------------------------------
# JSONC with positions - enough to change a value without touching the
# comments and layout around it
# --------------------------------------------------------------------------

class Node:
    def __init__(self, start):
        self.start, self.end = start, start
        self.value = None
        self.members = []       # objects: (key, value Node, key start)

    def get(self, key):
        return next((n for k, n, _ in self.members if k == key), None)


class _Parser:
    def __init__(self, text):
        self.t, self.i = text, 0

    def fail(self, what):
        line = self.t.count("\n", 0, self.i) + 1
        raise ValueError("%s at line %d" % (what, line))

    def skip(self):
        t = self.t
        while self.i < len(t):
            c = t[self.i]
            if c in " \t\r\n\ufeff":
                self.i += 1
            elif t.startswith("//", self.i):
                nl = t.find("\n", self.i)
                self.i = len(t) if nl < 0 else nl + 1
            elif t.startswith("/*", self.i):
                close = t.find("*/", self.i + 2)
                if close < 0:
                    self.fail("unclosed comment")
                self.i = close + 2
            else:
                return

    def value(self):
        self.skip()
        node, t = Node(self.i), self.t
        if self.i >= len(t):
            self.fail("unexpected end")
        c = t[self.i]
        if c == "{":
            self.i += 1
            node.value = {}
            while True:
                self.skip()
                if t.startswith("}", self.i):
                    self.i += 1
                    break
                key_start = self.i
                key = self.string()
                self.skip()
                if not t.startswith(":", self.i):
                    self.fail("expected ':'")
                self.i += 1
                child = self.value()
                node.members.append((key, child, key_start))
                node.value[key] = child.value
                self.skip()
                if t.startswith(",", self.i):
                    self.i += 1
                elif not t.startswith("}", self.i):
                    self.fail("expected ',' or '}'")
        elif c == "[":
            self.i += 1
            node.value = []
            while True:
                self.skip()
                if t.startswith("]", self.i):
                    self.i += 1
                    break
                node.value.append(self.value().value)
                self.skip()
                if t.startswith(",", self.i):
                    self.i += 1
                elif not t.startswith("]", self.i):
                    self.fail("expected ',' or ']'")
        elif c == '"':
            node.value = self.string()
        else:
            m = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null"
                           ).match(t, self.i)
            if not m:
                self.fail("unexpected %r" % c)
            node.value = json.loads(m.group(0))
            self.i = m.end()
        node.end = self.i
        return node

    def string(self):
        m = re.compile(r'"(?:[^"\\]|\\.)*"', re.S).match(self.t, self.i)
        if not m:
            self.fail("expected a string")
        self.i = m.end()
        return json.loads(m.group(0))


def parse(text):
    p = _Parser(text)
    root = p.value()
    p.skip()
    if p.i != len(text):
        p.fail("trailing text")
    return root


def _indent_at(text, pos):
    line_start = text.rfind("\n", 0, pos) + 1
    return re.match(r"[ \t]*", text[line_start:]).group(0)


def _render(value, indent):
    return json.dumps(value, indent=2).replace("\n", "\n" + indent)


# --------------------------------------------------------------------------
# working out the changes
# --------------------------------------------------------------------------

def variants_for(model_id, spec):
    """The variants block opencode should have for this model."""
    out = {level: {"chat_template_kwargs": spec.kwargs[level]}
           for level in spec.levels}
    lowered = model_id.lower()
    if spec.thinks and not any(s in lowered for s in OPENCODE_NO_GUESS):
        # Where a guess shares a level's name (Muse Glimmer's low / medium /
        # high) opencode merges the two and also sends reasoning_effort with
        # the same value - which llama.cpp hands the template as
        # reasoning_strength too, so it agrees with the kwargs.
        for guess in OPENCODE_GUESSES:
            out.setdefault(guess, {"disabled": True})
    return out


# What opencode sends per request, and what Zoomies calls the same setting.
# Anything else in an "options" block - a timeout, a header - is none of our
# business and is left exactly as written.
SAMPLING_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "seed": "seed",
}
# The same settings under the names the ai-sdk spells them, so a block
# written by hand in camelCase is recognised rather than duplicated.
_OPTION_ALIASES = {"topP": "top_p", "topK": "top_k", "minP": "min_p",
                   "repeatPenalty": "repeat_penalty",
                   "repetition_penalty": "repeat_penalty",
                   "presencePenalty": "presence_penalty"}


def _sampling_for_config(model):
    """The sampling numbers Zoomies owns for this model, or (None, why).

    They come from the model's presets, which is the only place Zoomies
    keeps numbers when the GUI is not open. Presets that disagree are the
    one case left alone: there is no single answer to write.

    A preset with no sampling numbers still counts as an answer - opencode
    is stripped back to the server's own values - because a model whose
    numbers live in two places is exactly what this is meant to end. The
    summary says so per model, loudly enough to act on: the fix is to save
    the numbers into the preset, not to leave them in opencode.
    """
    presets = list(state.presets_for(
        [c for c in (model.id, model.gguf_path, model.label) if c], "llamacpp"))
    subsets = []
    for preset in presets:
        settings = preset.get("settings") or {}
        subsets.append({k: settings[k] for k in SAMPLING_OPTIONS
                        if settings.get(k) not in (None, "")})
    for other in subsets[1:]:
        if other != subsets[0]:
            return None, ("its presets disagree about the sampling numbers "
                          "(%s vs %s)." % (subsets[0] or "none", other or "none"))
    return (subsets[0] if subsets else {}), ""


def _plan_options(plan, name, mnode, anchor, indent, wanted):
    """Make opencode's options block say what Zoomies says, and nothing else.

    opencode sends these with every request, and a request's own value beats
    whatever the server was started with - so a number left here quietly
    overrules Zoomies. Where Zoomies has a number, it is written; where it
    has none, the setting is removed and the server's own value stands.

    wanted is None when Zoomies cannot say what the numbers should be; then
    the block is not touched at all.
    """
    if wanted is None:
        return []
    o_node = mnode.get("options")
    old = dict((o_node.value if o_node else None) or {})
    keep = {k: v for k, v in old.items()
            if _OPTION_ALIASES.get(k, k) not in SAMPLING_OPTIONS}
    had = {_OPTION_ALIASES.get(k, k): v for k, v in old.items()
           if _OPTION_ALIASES.get(k, k) in SAMPLING_OPTIONS}
    new_options = dict(keep)
    new_options.update({SAMPLING_OPTIONS[k]: v for k, v in wanted.items()})
    if new_options == old:
        return []

    if not new_options:
        _remove_member(plan, mnode, "options")
    elif o_node is not None:
        plan.edits.append((o_node.start, o_node.end,
                           _render(new_options, indent)))
    else:
        plan.edits.append((anchor.end, anchor.end, ',\n%s"options": %s'
                           % (indent, _render(new_options, indent))))

    what = []
    for key in SAMPLING_OPTIONS:
        before, after = had.get(key), wanted.get(key)
        if before == after:
            continue
        if after is None:
            what.append("%s %s removed" % (key, json.dumps(before)))
        elif before is None:
            what.append("%s %s" % (key, json.dumps(after)))
        else:
            what.append("%s %s -> %s" % (key, json.dumps(before),
                                         json.dumps(after)))
    return ["options: " + ", ".join(what)] if what else []


def _limit_edits(plan, mnode, anchor, indent, context, max_tokens):
    """Tell opencode how much room the server actually has.

    opencode never asks: it reads these two numbers from its own config and
    trusts them. "context" is what it thinks the window is, and "output" is
    both the max_tokens it sends and the room it holds back - it starts
    compacting at context minus output. Left stale, a 64k number against a
    262k server compacts at a quarter of the room there really is.

    Only these two members are touched. A "limit" block can also carry an
    "input", which opencode prefers for the compaction sum when it is there;
    anything already written is kept.
    """
    wanted = {}
    if context:
        wanted["context"] = int(context)
    if max_tokens:
        wanted["output"] = int(max_tokens)
    if not wanted:
        return []
    l_node = mnode.get("limit")
    old = dict((l_node.value if l_node else None) or {})
    what = ["%s %s -> %s" % (k, json.dumps(old.get(k)), json.dumps(v))
            for k, v in wanted.items() if old.get(k) != v]
    if not what:
        return []
    merged = dict(old)
    merged.update(wanted)
    if l_node is None:
        plan.edits.append((anchor.end, anchor.end, ',\n%s"limit": %s'
                           % (indent, _render(merged, indent))))
    else:
        plan.edits.append((l_node.start, l_node.end, _render(merged, indent)))
    return ["limit: " + ", ".join(what)]


def _preset_max_tokens(candidates):
    """What the presets say a reply may run to, or "" if they cannot agree."""
    values = set()
    for preset in state.presets_for([c for c in candidates if c], "llamacpp"):
        value = (preset.get("settings") or {}).get("max_tokens")
        if value not in (None, ""):
            values.add(str(value).strip())
    return values.pop() if len(values) == 1 else ""


def _preset_context(candidates):
    """What the presets say the context is, or 0 if they cannot agree."""
    values = set()
    for preset in state.presets_for([c for c in candidates if c], "llamacpp"):
        value = (preset.get("settings") or {}).get("context_length")
        if value not in (None, ""):
            values.add(str(value).strip())
    if len(values) != 1:
        return 0
    try:
        return int(values.pop())
    except ValueError:
        return 0


def _spec_for_config(model):
    """The Spec across every llama.cpp preset for this model. Presets can
    name different --chat-template-file files; if those disagree, opencode
    cannot know which server it is talking to."""
    presets = list(state.presets_for(
        [c for c in (model.id, model.gguf_path, model.label) if c], "llamacpp"))
    flags = {str((p.get("settings") or {}).get("extra_flags") or "")
             for p in presets} or {""}
    specs = [reasoning.spec_for(model, f) for f in sorted(flags)]
    first = specs[0]
    for other in specs[1:]:
        if (other.levels, other.kwargs, other.thinks, other.problem) != \
                (first.levels, first.kwargs, first.thinks, first.problem):
            return reasoning.Spec(problem=(
                "Its presets use different chat templates (%s vs %s), which "
                "accept different levels." % (first.source, other.source)))
    return first


class Plan:
    def __init__(self, path, text):
        self.path, self.text = path, text
        self.edits = []         # (start, end, replacement)
        self.changed = []       # "provider/model: what"
        self.skipped = []       # "provider/model: why"
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
            parts.append("Will change:\n" + "\n".join("  " + c for c in self.changed))
        else:
            parts.append("Nothing to change - opencode already matches.")
        if self.skipped:
            parts.append("Left alone:\n" + "\n".join("  " + s for s in self.skipped))
        if self.notes:
            parts.append("Notes:\n" + "\n".join("  " + n for n in self.notes))
        return "\n\n".join(parts)


def plan_sync(path=None, models=None, loaded=None):
    """Read opencode's config and work out the edits; writes nothing."""
    path = path or config_path()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    root = parse(text)
    plan = Plan(path, text)
    if models is None:
        models = backends.get("llamacpp").list_models()
    # A single-model llama-server answers to any name, so the config may use
    # a loose file's name (Qwen3.6-35B-A3B-UD-Q4_K_XL) rather than its path.
    by_id = {m.label: m for m in models if m.source == "folder"}
    by_id.update({m.id: m for m in models})
    live = _live_by_port(loaded)
    providers = root.get("provider")
    if providers is None:
        plan.notes.append("No providers in %s." % path)
        return plan

    new_variants = {}                 # "provider/model" -> (old, new) names
    for pid, pnode, _ in providers.members:
        base = str(((pnode.value or {}).get("options") or {}).get("baseURL") or "")
        m = _LOCAL_URL.match(base)
        port = int(m.group(1)) if m else 0
        if not (backends.LLAMACPP_PORT <= port < backends.LLAMACPP_PORT + 50):
            continue
        models_node = pnode.get("models")
        for mid, mnode, key_start in (models_node.members if models_node else []):
            name = "%s/%s" % (pid, mid)
            model = by_id.get(mid)
            if model is None:
                plan.skipped.append("%s: not a model Zoomies finds for "
                                    "llama.cpp." % name)
                continue
            spec = _spec_for_config(model)
            if spec.problem:
                plan.skipped.append("%s: %s" % (name, spec.problem))
                continue
            for note in spec.notes:
                plan.notes.append("%s: %s" % (mid, note))
            sampling, why = _sampling_for_config(model)
            if sampling is None:
                # No single answer to write, so its options block is left as
                # it is - but the reasoning half of the sync still applies.
                plan.skipped.append("%s: options left alone, %s" % (name, why))
            elif not sampling:
                plan.notes.append(
                    "%s: no preset of its own holds sampling numbers, so "
                    "opencode is left sending none and the server's own "
                    "values stand. Save them into its preset to pin them."
                    % name)
            # What a server is serving beats what a preset asked for: the
            # context can be typed into the form without saving, and llama.cpp
            # can hand out less than was asked for. Only a model that is down
            # is described by its preset.
            running = live.get((port, mid))
            context = (running.context if running else
                       _preset_context([model.id, model.gguf_path, model.label]))
            limits = (context, backends.max_tokens_for(
                context, _preset_max_tokens(
                    [model.id, model.gguf_path, model.label])))
            _plan_model(plan, name, mid, mnode, spec, new_variants, sampling,
                        limits)

    _plan_agents(plan, root, new_variants)
    return plan


def _live_by_port(loaded=None):
    """{(port, name): model} for every llama.cpp server that is up."""
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
    """Edits that put the running servers' real context into opencode.

    The context a model is serving is decided at load time and changes from
    one load to the next, so it is read back off the server rather than
    taken from a preset: llama.cpp can hand out less than was asked for, and
    what it handed out is what opencode has to believe.

    Only models that are up are touched. An entry for something not loaded
    keeps whatever it has - it says nothing about a server that isn't there.
    """
    path = path or config_path()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    root = parse(text)
    plan = Plan(path, text)
    live = _live_by_port(loaded)
    providers = root.get("provider")
    if providers is None or not live:
        return plan

    for pid, pnode, _ in providers.members:
        base = str(((pnode.value or {}).get("options") or {}).get("baseURL") or "")
        match = _LOCAL_URL.match(base)
        if not match:
            continue
        port = int(match.group(1))
        models_node = pnode.get("models")
        for mid, mnode, key_start in (models_node.members if models_node else []):
            model = live.get((port, mid))
            if model is None or not mnode.members:
                continue
            indent = _indent_at(text, mnode.members[0][2])
            anchor = mnode.members[0][1]
            what = _limit_edits(
                plan, mnode, anchor, indent, model.context,
                backends.max_tokens_for(
                    model.context, _preset_max_tokens([model.id, model.label])))
            if what:
                plan.changed.append("%s/%s: %s" % (pid, mid, "; ".join(what)))
    return plan


def _plan_model(plan, name, mid, mnode, spec, new_variants, sampling=None,
                limits=(0, 0)):
    text = plan.text
    want_reason = bool(spec.thinks)
    want_variants = variants_for(mid, spec)
    old = mnode.value or {}
    what = []

    r_node, v_node = mnode.get("reasoning"), mnode.get("variants")
    anchor = r_node or (mnode.members[0][1] if mnode.members else None)
    if anchor is None:
        plan.skipped.append("%s: empty entry." % name)
        return
    indent = _indent_at(text, (mnode.members[0][2]))

    if r_node is None:
        plan.edits.append((anchor.end, anchor.end, ',\n%s"reasoning": %s'
                           % (indent, json.dumps(want_reason))))
        what.append("reasoning %s" % json.dumps(want_reason))
    elif old.get("reasoning") is not want_reason:
        plan.edits.append((r_node.start, r_node.end, json.dumps(want_reason)))
        what.append("reasoning %s -> %s" % (json.dumps(old.get("reasoning")),
                                             json.dumps(want_reason)))

    old_variants = old.get("variants") or {}
    if want_variants != old_variants:
        if not want_variants:
            _remove_member(plan, mnode, "variants")
        elif v_node is not None:
            plan.edits.append((v_node.start, v_node.end,
                               _render(want_variants, indent)))
        else:
            plan.edits.append((anchor.end, anchor.end, ',\n%s"variants": %s'
                               % (indent, _render(want_variants, indent))))
        shown = [k for k, v in want_variants.items() if not v.get("disabled")]
        what.append("variants %s" % (" / ".join(shown) or "none"))

    what += _limit_edits(plan, mnode, anchor, indent, *limits)
    what += _plan_options(plan, name, mnode, anchor, indent, sampling)
    if what:
        plan.changed.append("%s: %s" % (name, "; ".join(what)))
    new_variants[name] = (old_variants, want_variants)


def _remove_member(plan, obj, key):
    text = plan.text
    for i, (k, node, key_start) in enumerate(obj.members):
        if k != key:
            continue
        if i + 1 < len(obj.members):         # up to the next key
            start, end = key_start, obj.members[i + 1][2]
        elif i > 0:                          # from the previous value
            start, end = obj.members[i - 1][1].end, node.end
        else:                                # the only member, and its comma
            comma = re.match(r"\s*,", text[node.end:])
            start, end = key_start, node.end + (comma.end() if comma else 0)
        plan.edits.append((start, end, ""))
        return


def _plan_agents(plan, root, new_variants):
    agents = root.get("agent")
    for aname, anode, _ in (agents.members if agents else []):
        cfg = anode.value or {}
        target, chosen = cfg.get("model"), cfg.get("variant")
        if not chosen or target not in new_variants:
            continue
        old, new = new_variants[target]
        live = [k for k, v in new.items() if not v.get("disabled")]
        if chosen in live:
            continue
        before = ((old.get(chosen) or {}).get("chat_template_kwargs") or {})
        match = next((k for k in live if before and before.items() <= (
            new[k].get("chat_template_kwargs") or {}).items()), None)
        if match:
            vnode = anode.get("variant")
            plan.edits.append((vnode.start, vnode.end, json.dumps(match)))
            plan.changed.append("agent %s: variant %s -> %s (sends the same "
                                "settings)" % (aname, chosen, match))
        else:
            plan.skipped.append("agent %s: its variant %r is not one of %s's "
                                "levels (%s) - pick one." % (
                                    aname, chosen, target, " / ".join(live)))


def write(plan, backup=True):
    """Back up the config, then write the planned text. Returns the backup.

    backup=False keeps one rolling copy instead of a dated one: the sync
    that runs on every load would otherwise fill the folder with them.
    """
    new_text = plan.new_text
    parse(new_text)                   # never write something opencode can't read
    backup = ("%s.bak-%s" % (plan.path, time.strftime("%Y%m%d-%H%M%S"))
              if backup else plan.path + ".bak-auto")
    with open(backup, "w", encoding="utf-8", newline="") as f:
        f.write(plan.text)
    tmp = plan.path + ".zoomies-tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    os.replace(tmp, plan.path)
    return backup
