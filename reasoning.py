r"""
Zoomies - what a model's reasoning switch actually accepts.

Every model does this differently, and the only place that cannot be out of
date is the chat template the server renders - the jinja inside the .gguf, or
the file an Extra flags --chat-template-file points at. So this reads the
template and reports the levels it understands:

    Qwen3.6, Ornith, Gemma 4, Laguna   enable_thinking            off / on
    Qwen3.8                            enable_thinking + reasoning_effort
                                       off / low / medium / xhigh
    Granite 4.2                        enable_thinking + low_effort
                                       off / low-effort / on
    Muse Glimmer                       reasoning_strength, no off switch
                                       low / medium / high / xhigh
    Ministral 3, Devstral Small 2      nothing to switch

Each level is the exact chat_template_kwargs that produce it, and it is the
same dict whether Zoomies bakes it into the server (--chat-template-kwargs) or
opencode sends it with a request. Every level names every variable it relies
on: llama.cpp merges a request's kwargs into the server's key by key, so a
level that left enable_thinking out would inherit "off" from the launch.

Where the template reads a value but does not list the allowed ones (Muse
Glimmer takes any string), the values come from DOCUMENTED below, which cites
where each list was read. Anything else it cannot account for is reported as
a problem or a note - never filled in with a plausible-looking default.
"""

import json
import os
import re
import shlex
import struct
from dataclasses import dataclass, field

import gguf

# How llama.cpp treats a template it has not been told anything about: with
# --reasoning left on "auto" and --jinja, it passes enable_thinking=true to
# every template that reacts to it (server-context.cpp). So Gemma 4 and
# Laguna, whose templates default to off, still think under llama.cpp.
SWITCH = "enable_thinking"

# String-valued knobs. llama.cpp copies a request's reasoning_effort into both
# of these template variables (common/jinja/caps.cpp).
VALUE_VARS = ("reasoning_effort", "reasoning_strength")

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Values a template reads but does not enumerate, and boolean effort flags,
# taken from the model's own documentation. Keyed by a fragment of the model
# name with the punctuation removed ("granite4.2:30b" -> "granite42...").
DOCUMENTED = (
    {"match": "museglimmer", "var": "reasoning_strength",
     "values": ("low", "medium", "high", "xhigh"),
     "source": "Unsloth's Muse Glimmer guide, Thinking Settings"},
    {"match": "granite4", "flag": "low_effort", "level": "low-effort",
     "source": "IBM's Granite 4.2 model card, Thinking Modes"},
)

# Template variables about thinking that are not a level: whether old
# reasoning is kept in the history, for instance.
_NOT_A_LEVEL = re.compile(r"history|preserve|keep|clear|truncate")

_FILE_FLAGS = ("--chat-template-file",)
_INLINE_FLAGS = ("--chat-template",)


@dataclass
class Spec:
    levels: tuple = ()            # least reasoning first
    kwargs: dict = field(default_factory=dict)    # level -> template kwargs
    default: str = ""             # what the server does when told nothing
    off: str = ""                 # the level that switches thinking off
    thinks: bool = False          # reasons at all, switchable or not
    source: str = ""              # where the template came from
    notes: tuple = ()             # things read but deliberately not offered
    problem: str = ""             # why no levels can be offered

    @property
    def usable(self):
        return bool(self.levels) and not self.problem

    def describe(self):
        if self.problem:
            return self.problem
        if not self.levels:
            return ("Always reasons; the template has no switch." if self.thinks
                    else "No reasoning to switch.")
        return "%s (default %s), from the %s." % (
            " / ".join(self.levels), self.default, self.source)


# --------------------------------------------------------------------------
# finding the template
# --------------------------------------------------------------------------

def _flag_value(tokens, names):
    for i, tok in enumerate(tokens):
        head, eq, tail = tok.partition("=")
        if head in names:
            value = tail if eq else (tokens[i + 1] if i + 1 < len(tokens) else "")
            return value.strip().strip("'\"")
    return None


def split_flags(extra_flags):
    raw = str(extra_flags or "").strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=False)
    except ValueError:
        return raw.split()


def template_for(model, extra_flags=""):
    """(template text, where it came from) or (None, why not)."""
    tokens = split_flags(extra_flags)
    path = _flag_value(tokens, _FILE_FLAGS)
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read(), "template file %s" % os.path.basename(path)
        except OSError as exc:
            return None, "Could not read the template file %s: %s" % (path, exc)
    if _flag_value(tokens, _INLINE_FLAGS) is not None:
        return None, ("Extra flags choose a built-in llama.cpp template "
                      "(--chat-template), which Zoomies cannot read.")
    model_file = getattr(model, "gguf_path", "") or ""
    if not model_file:
        return None, "No .gguf file to read the chat template from."
    try:
        text = gguf.read(model_file).chat_template
    except (OSError, ValueError, struct.error) as exc:
        return None, "Could not read %s: %s" % (os.path.basename(model_file), exc)
    if not text:
        return None, ("%s has no chat template inside it, and Extra flags do "
                      "not name one (--chat-template-file)."
                      % os.path.basename(model_file))
    return text, "chat template inside %s" % os.path.basename(model_file)


# --------------------------------------------------------------------------
# reading the template
# --------------------------------------------------------------------------

def _reads(text, var):
    """The template uses var as a variable, not just inside a string."""
    return re.search(r"(?<![\w.'\"])%s\b" % re.escape(var), text) is not None


def _literals(chunk):
    return re.findall(r"['\"]([A-Za-z0-9_-]+)['\"]", chunk)


def _allowed_values(text, var):
    """Values a template checks var against: `x not in ('a', 'b')`, the way
    Qwen3.8 rejects anything outside xhigh/medium/low."""
    m = re.search(r"\w*%s\s+(?:not\s+)?in\s*[\(\[]([^\)\]]*)[\)\]]"
                  % re.escape(var), text)
    return _literals(m.group(1)) if m else []


def _default_value(text, var):
    v = re.escape(var)
    for pattern in (r"%s\s*\|\s*default\(\s*['\"](\w+)['\"]" % v,
                    r"%s\s+if\s+%s\s+is\s+defined[^%%}]*?else\s*['\"](\w+)['\"]"
                    % (v, v)):
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return ""


def _effort_flags(text):
    """Boolean flags like `set low_effort = low_effort if low_effort is
    defined else False` - Granite's low_effort, Nemotron's medium_effort."""
    return re.findall(r"set\s+(\w+_effort)\s*=\s*\1\s+if\s+\1\s+is\s+defined"
                      r"\s+else\s+False", text)


def _other_knobs(text, handled):
    """Thinking-related variables the template reads that nothing here
    handles, so they can be reported instead of silently ignored."""
    names = set(re.findall(r"(?<![\w.])(\w+)\s+is\s+(?:un)?defined", text))
    names |= set(re.findall(r"(?<![\w.])(\w+)\s*\|\s*default\(", text))
    return sorted(n for n in names
                  if re.search(r"think|reason|effort", n)
                  and n not in handled and not _NOT_A_LEVEL.search(n)
                  and not n.startswith(("resolved_", "message", "reasoning_content")))


def _documented(model_id):
    flat = re.sub(r"[^a-z0-9]", "", str(model_id).lower())
    return [d for d in DOCUMENTED if d["match"] in flat]


def _server_forces_off(tokens):
    value = _flag_value(tokens, ("-rea", "--reasoning"))
    budget = _flag_value(tokens, ("--reasoning-budget",))
    return value == "off" or budget == "0"


def _ordered(values):
    known = [v for v in EFFORT_ORDER if v in values]
    return known + [v for v in values if v not in known]


def detect(text, model_id="", extra_flags="", source="chat template"):
    """A Spec from template text. Pure: no files, so it can be tested."""
    tokens = split_flags(extra_flags)
    docs = _documented(model_id)
    notes, handled = [], {SWITCH}
    switch = _reads(text, SWITCH)

    # String-valued knob: the template's own list first, the docs second.
    value_var, values, default_value = "", [], ""
    for var in VALUE_VARS:
        if not _reads(text, var):
            continue
        handled.add(var)
        listed = _allowed_values(text, var)
        doc = next((d for d in docs if d.get("var") == var), None)
        flag_alias = re.search(r"set\s+(\w+_effort)\s*=\s*%s\s*==" % var, text)
        if listed:
            found = listed
        elif doc:
            found = list(doc["values"])
            notes.append("%s values from %s." % (var, doc["source"]))
        elif flag_alias:
            # Granite: reasoning_effort only feeds low_effort, handled below.
            continue
        else:
            return Spec(source=source, thinks=True, problem=(
                "The %s reads %s but does not list the values it accepts, and "
                "no documented list is on file for this model."
                % (source, var)))
        if value_var:
            notes.append("Also reads %s; only %s is offered." % (var, value_var))
            continue
        value_var = var
        values = _ordered([v for v in found if not (switch and v == "none")])
        default_value = _default_value(text, var)
        if default_value not in values:
            return Spec(source=source, thinks=True, problem=(
                "The %s reads %s (%s) but does not say which one it uses by "
                "default." % (source, var, ", ".join(values))))

    # Boolean effort flags: offered only when the model's docs describe them.
    flags = []
    for flag in _effort_flags(text):
        handled.add(flag)
        doc = next((d for d in docs if d.get("flag") == flag), None)
        if doc:
            flags.append((flag, doc["level"]))
            notes.append("%s from %s." % (doc["level"], doc["source"]))
        else:
            notes.append("The template also has a %s switch that the model's "
                         "docs do not describe, so it is not offered." % flag)

    for knob in _other_knobs(text, handled):
        notes.append("The template also reads %s, which is not offered." % knob)

    if not switch and not value_var and not flags:
        return Spec(source=source, thinks="<think>" in text, notes=tuple(notes))

    on = {SWITCH: True} if switch else {}
    quiet = {name: False for name, _ in flags}
    levels, kwargs = [], {}

    def add(name, kw):
        levels.append(name)
        kwargs[name] = kw

    if switch:
        add("off", {SWITCH: False})
    for name, label in flags:
        add(label, dict(on, **dict(quiet, **{name: True})))
    if value_var:
        for v in values:
            add(v, dict(on, **dict(quiet, **{value_var: v})))
        default = default_value
    else:
        add("on", dict(on, **quiet))
        default = "on"
    if switch and _server_forces_off(tokens):
        default = "off"
        notes.append("Extra flags switch reasoning off on the server.")

    off = "off" if switch else ("none" if "none" in levels else "")
    return Spec(levels=tuple(levels), kwargs=kwargs, default=default, off=off,
                thinks=True, source=source, notes=tuple(notes))


def spec_for(model, extra_flags=""):
    """The reasoning Spec for a model as it would be launched."""
    text, where = template_for(model, extra_flags)
    if text is None:
        return Spec(problem=where)
    return detect(text, getattr(model, "id", ""), extra_flags, source=where)


def kwargs_json(kwargs):
    return json.dumps(kwargs, separators=(",", ":"))
