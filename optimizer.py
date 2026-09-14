r"""
Zoomies - reading recommended settings out of the live Unsloth docs.

The whole design rests on one discovery: unsloth.ai publishes
https://unsloth.ai/docs/llms.txt, a 32 KB machine-readable index of every
documentation page with its canonical URL. And the URL slug IS the model
family name:

    qwen3.8:27b-q4_K_M    ->  qwen3.8     ->  /docs/models/qwen3.8.md
    gemma4:12b-it-q4_K_M  ->  gemma4      ->  /docs/models/gemma-4.md
    ministral-3:14b-...   ->  ministral3  ->  /docs/models/tutorials/ministral-3.md

Take the Ollama name up to the colon, drop punctuation, and it matches the
slug with punctuation dropped. That is the entire lookup.

An earlier version of this module scanned the 1.9 MB llms-full.txt corpus
and guessed where each page started and stopped. That approach needed code
fence tracking, an all-caps filter to stop prompt-template comments being
read as page titles, and a heuristic for "does this heading look real" -
and it still got Ministral's page boundary wrong. Fetching the one page you
actually want makes all of that unnecessary: the file IS the page.

The governing rule is unchanged:

    FAIL TO EMPTY, NEVER TO WRONG.

An empty box is obviously empty. A confidently wrong temperature looks
exactly like a right one.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import backends
import state

INDEX_URL = "https://unsloth.ai/docs/llms.txt"
INDEX_PATH = os.path.join(state.CACHE_DIR, "llms.txt")
PAGE_DIR = os.path.join(state.CACHE_DIR, "pages")
META_PATH = os.path.join(state.CACHE_DIR, "docs-meta.json")

TTL_SECONDS = 24 * 3600

# Docs routinely quote a context this hardware cannot reach. 262,144 tokens
# of KV cache will not fit beside the weights on a 16 GB card, and silently
# filling that in would make the app look broken on the very first click.
SAFE_CONTEXT = 32768

USER_AGENT = "Zoomies/1.0 (+local)"


# --------------------------------------------------------------------------
# text cleaning
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_SHORTCODE = re.compile(r"\{%[^%]*%\}")
_EMOJI_NAME = re.compile(r":[a-z0-9_+-]+:")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF←-⇿⌀-➿️☀-⛿]")
_ENTITY = re.compile(r"&#x[0-9A-Fa-f]+;|&[a-z]+;")


def clean(text):
    """Strip everything GitBook and Markdown put around the actual words."""
    t = str(text)
    t = _SHORTCODE.sub(" ", t)
    t = _TAG.sub(" ", t)
    t = _ENTITY.sub(" ", t)
    t = _EMOJI_NAME.sub(" ", t)
    t = _EMOJI.sub(" ", t)
    t = t.replace("\\_", "_").replace("\\*", "*").replace("\\-", "-")
    t = t.replace("\\|", "|").replace("`", "").replace("**", "").replace("*", "")
    return re.sub(r"\s+", " ", t).strip()


def norm_key(text):
    """Punctuation-free lowercase key. qwen3.8 and Qwen3.8 and qwen-3.8 all
    land on the same string; the dot survives because qwen3 and qwen3.8 are
    genuinely different models."""
    return re.sub(r"[^a-z0-9.]", "", str(text).lower())


# --------------------------------------------------------------------------
# the settings vocabulary
# --------------------------------------------------------------------------

ALIAS = {
    "temperature": "temperature", "temp": "temperature",
    "top_p": "top_p", "topp": "top_p", "nucleus": "top_p",
    "top_k": "top_k", "topk": "top_k",
    "min_p": "min_p", "minp": "min_p",
    "repetition_penalty": "repeat_penalty", "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
    "context_length": "context_length", "context_window": "context_length",
    "maximum_context_window": "context_length",
    "maximum_context_length": "context_length",
    "max_context_window": "context_length", "max_context_length": "context_length",
    "max_context": "context_length", "num_ctx": "context_length",
    "max_seq_length": "context_length", "ctx_size": "context_length",
    "context": "context_length",
}

_NAMES = (
    r"temperature|temp|top[_\-\s]?p|top[_\-\s]?k|min[_\-\s]?p|nucleus|"
    r"repetition[_\-\s]?penalty|repeat[_\-\s]?penalty|repeat[_\-\s]?last[_\-\s]?n|"
    r"presence[_\-\s]?penalty|frequency[_\-\s]?penalty|seed|"
    r"max(?:imum)?[_\-\s]?context(?:[_\-\s]?(?:window|length))?|"
    r"context[_\-\s]?(?:window|length)|num[_\-\s]?ctx|"
    r"max[_\-\s]?seq[_\-\s]?length|ctx[_\-\s]?size"
)

# A value must START with a digit. That single rule is what makes
# "Top_P = default" in the Ministral table correctly produce nothing rather
# than a zero.
ASSIGN = re.compile(
    r"(?<![A-Za-z0-9_])(%s)\s*[=:]\s*([0-9][0-9,._]*)" % _NAMES, re.I)

CONTEXT_PHRASE = re.compile(
    r"(?i)(max(?:imum)?\s+context(?:\s+(?:window|length))?|"
    r"context\s+(?:window|length)|num_ctx|max_seq_length)")
# Numbers are collected separately rather than in one regex, because the text
# between the phrase and the value can itself contain digits:
#   "The maximum context length Ministral 3 can reach is `262,144`"
NUMBER = re.compile(r"([0-9][0-9,._]*)\s*([KM])?\b")

INT_KEYS = {"top_k", "seed", "context_length", "repeat_last_n"}


def canon(name):
    key = re.sub(r"[\s\-]+", "_", clean(name).lower().strip().rstrip(":"))
    key = re.sub(r"_+", "_", key).strip("_")
    return ALIAS.get(key)


def to_number(raw, key=None):
    txt = str(raw).replace(",", "").rstrip(".")
    if not txt:
        return None
    try:
        value = float(txt)
    except ValueError:
        return None
    if key in INT_KEYS or value.is_integer() and key not in (
            "temperature", "top_p", "min_p", "repeat_penalty",
            "presence_penalty", "frequency_penalty"):
        return int(value)
    return value


def fmt(value):
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return "%.1f" % value
    return str(value)


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def _age_text(seconds):
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return "%d min ago" % round(seconds / 60)
    if seconds < 36 * 3600:
        return "%d hours ago" % round(seconds / 3600)
    return "%d days ago" % round(seconds / 86400)


def _meta():
    return state.read_json(META_PATH, {})


def _stamp(key):
    meta = _meta()
    meta[key] = time.time()
    state.write_json(META_PATH, meta)


def _download(url, path, min_bytes=2000):
    """Download to a temp file then rename, so an interrupted fetch can never
    leave a truncated file that later parses as an empty page."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=45) as resp:
        blob = resp.read()
    if len(blob) < min_bytes:
        raise ValueError("response was only %d bytes" % len(blob))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(blob)
    os.replace(tmp, path)
    return blob.decode("utf-8", "replace")


def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _cached(path, key, url, force=False, min_bytes=2000):
    """Return (text, age_label, error).

    The offline ladder matters more than the happy path: a stale cache is
    used rather than blanked, because a flaky network is not a reason to
    throw away working data.
    """
    state.ensure_dirs()
    stamped = float(_meta().get(key) or 0)
    age = time.time() - stamped if stamped else None
    have = os.path.isfile(path)

    if have and age is not None and age < TTL_SECONDS and not force:
        return _read(path), _age_text(age), ""
    try:
        text = _download(url, path, min_bytes)
        _stamp(key)
        return text, "just now", ""
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = str(getattr(exc, "reason", exc))
        if have:
            return (_read(path),
                    "%s - offline" % _age_text(age if age is not None else 0), "")
        return "", "", reason


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------

LINK = re.compile(r"\[([^\]]+)\]\((https://unsloth\.ai/docs/models/[^)]+)\)")

# Sub-pages of a model that are not "how to run it" pages.
_NOT_A_MODEL = {"train", "qat", "finetune", "finetuning", "ggufbenchmarks",
                "tutorials", "mtp", "howto", "benchmarks", "reinforcementlearning"}
# Suffixes the docs bolt onto a slug that are not part of the model name.
_SLUG_TAIL = re.compile(r"-(how-to-run|run-locally|how-to|tutorial|guide).*$")


def page_family(url):
    """Family key from a docs URL. This is the naming convention."""
    tail = url.rsplit("/", 1)[-1]
    if tail.endswith(".md"):
        tail = tail[:-3]
    return norm_key(_SLUG_TAIL.sub("", tail))


def model_family(model_id):
    """Family key from a backend's model name - everything before the tag."""
    return norm_key(str(model_id).split(":")[0])


def fetch_index(force=False):
    """Return ({family: {title, url}}, age_label, error)."""
    text, age, err = _cached(INDEX_PATH, "index", INDEX_URL, force, min_bytes=2000)
    if not text:
        return {}, age, err
    index = {}
    for title, url in LINK.findall(text):
        key = page_family(url)
        if not key or key in _NOT_A_MODEL:
            continue
        index.setdefault(key, {"title": title.strip(), "url": url, "family": key})
    return index, age, err


def fetch_page(entry, force=False):
    """Return (text, age_label, error) for one model page."""
    path = os.path.join(PAGE_DIR, entry["family"] + ".md")
    return _cached(path, "page:" + entry["family"], entry["url"], force,
                   min_bytes=500)


# --------------------------------------------------------------------------
# family / version matching
# --------------------------------------------------------------------------

VERSIONED = re.compile(r"^([a-z]+)([0-9]+(?:\.[0-9]+)*)(.*)$")


def _split_version(key):
    m = VERSIONED.match(key)
    if not m:
        return key, None, ""
    try:
        parts = [int(p) for p in m.group(2).split(".")]
    except ValueError:
        return key, None, ""
    return m.group(1), parts, m.group(3)


def previous_version(index, family):
    """The most recent earlier release of the same family.

    gemma4 has a page, but if it ever did not, gemma3 is a far better answer
    than nothing - the sampling defaults of a model family rarely swing
    wildly between versions. The caller must tell the user this happened.
    """
    stem, version, suffix = _split_version(family)
    if version is None:
        return None
    siblings = []
    for key, entry in index.items():
        s, v, suf = _split_version(key)
        if s == stem and suf == suffix and v is not None and v < version:
            siblings.append((v, entry))
    if not siblings:
        return None
    return max(siblings, key=lambda pair: pair[0])[1]


def find_page(index, family, manual_title=""):
    """Return (entry, how). Exact slug, an explicit manual choice, or the
    previous version of the same family - never a fuzzy neighbour.

    "qwen3" must never resolve to "qwen3.8" and "gemma4" must never silently
    become "gemma3"; those are different models with different sampling.
    """
    if manual_title:
        for entry in index.values():
            if entry["title"] == manual_title:
                return entry, "manual"
    if family in index:
        return index[family], "exact"
    older = previous_version(index, family)
    if older is not None:
        return older, "older"
    return None, "none"


# --------------------------------------------------------------------------
# locating the settings inside a page
# --------------------------------------------------------------------------

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Ranked, not a flat alternation. The Qwen3.8 page has BOTH a "Usage Guide"
# (which holds hardware requirements) and a "Recommended Settings" further
# down; taking whichever came first read the requirements tables and turned
# VRAM figures into sampling values. Ministral 3 has only "Usage Guide", so
# the weak pattern still has to exist - just never in preference.
SETTINGS_HDR = (
    (3, re.compile(r"(?i)(recommended\s+settings|official\s+recommended|"
                   r"recommended\s+parameters)")),
    (2, re.compile(r"(?i)(inference\s+settings|generation\s+settings|"
                   r"sampling\s+settings|recommended\s+inference)")),
    (1, re.compile(r"(?i)usage\s+guide")),
)


def find_settings_block(lines):
    """Return (start, end, heading) for the best-ranked settings section."""
    best = None
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if not m:
            continue
        depth, title = len(m.group(1)), clean(m.group(2))
        rank = 0
        for score, pattern in SETTINGS_HDR:
            if pattern.search(title):
                rank = score
                break
        if not rank or (best and rank <= best[0]):
            continue
        end = len(lines)
        for j in range(i + 1, len(lines)):
            m2 = HEADING.match(lines[j])
            if m2 and len(m2.group(1)) <= depth:
                end = j
                break
        best = (rank, i, end, title)
    return best[1:] if best else None


SIZE_TOKEN = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*([BTM])\b")


def size_tokens(text):
    return {"%s%s" % (a, b.lower()) for a, b in SIZE_TOKEN.findall(str(text))}


def _billions(token):
    m = re.match(r"^([0-9.]+)([btm])$", token)
    if not m:
        return None
    value = float(m.group(1))
    return {"m": value / 1000.0, "b": value, "t": value * 1000.0}[m.group(2)]


_NOT_SETTINGS = re.compile(r"(?i)(requirement|chat\s*template|tutorial|install|"
                           r"download|deployment|benchmark|quant)")


def find_variants(lines, start, end):
    """Sub-sections of the settings block, e.g. per model size."""
    subs = []
    base_depth = None
    for i in range(start, end):
        m = HEADING.match(lines[i])
        if not m:
            continue
        depth = len(m.group(1))
        if base_depth is None:
            base_depth = depth
            continue
        if depth <= base_depth:
            continue
        subs.append({"line": i, "title": clean(m.group(2)), "depth": depth})
    for a, b in zip(subs, subs[1:] + [{"line": end}]):
        a["end"] = b["line"]
    return subs


def pick_variant(variants, model):
    """Exact size match if there is one, otherwise the nearest size.

    The Qwen3.8 page offers 27B and 2.4T settings; a 27B model must get the
    27B block, and anything else should get whichever is closest rather than
    whichever happened to be written first. Sections that are plainly not
    settings (requirements, chat templates) are excluded first - matching
    "Qwen3.8-27B Requirements:" on its size token is how VRAM figures ended
    up in the temperature box.
    """
    usable = [v for v in variants if not _NOT_SETTINGS.search(v["title"])]
    sized = [(v, size_tokens(v["title"])) for v in usable]
    sized = [(v, t) for v, t in sized if t]
    if len(sized) < 2:
        return None

    want = size_tokens(model.id) | size_tokens(getattr(model, "size_hint", ""))
    for variant, tokens in sized:                      # exact
        if want & tokens:
            return variant

    target = next((b for b in (_billions(t) for t in sorted(want)) if b), None)
    if target is None:
        return None

    def distance(pair):                                # nearest by magnitude
        sizes = [b for b in (_billions(t) for t in pair[1]) if b]
        return min(abs(s - target) for s in sizes) if sizes else float("inf")

    nearest = min(sized, key=distance)
    return nearest[0] if distance(nearest) != float("inf") else None


# --------------------------------------------------------------------------
# parsing the numbers
# --------------------------------------------------------------------------

ROW = re.compile(r"^\s*\|(.+)\|\s*$")
RULE = re.compile(r"^\s*\|[\s:|\-]+\|\s*$")


def mode_key(label):
    """Collapse wildly inconsistent column labels onto a few stable keys.

    Order matters: "Instruct (non-thinking) Mode" contains the word
    "thinking", so non-thinking and instruct have to be tested first.
    """
    t = clean(label).lower().rstrip(":")
    t = re.sub(r"[^a-z0-9 ]+", " ", t).strip()
    if not t or t in ("default", "parameter", "value", "setting", "settings"):
        return "default"
    if "non thinking" in t or "nonthinking" in t or "instruct" in t or "chat" in t:
        return "instruct"
    if "think" in t or "reason" in t:
        return "thinking"
    if "coder" in t or "code" in t:
        return "coder"
    if "base" in t:
        return "base"
    return t[:24]


def _row_cells(line):
    m = ROW.match(line)
    if not m or RULE.match(line):
        return None
    return [clean(c) for c in m.group(1).split("|")]


def parse_tables(block_lines):
    """Two table shapes appear in these docs:

      P  | `temperature` | 1.0 | 0.7 |     parameter in column 0, modes in
                                            the header, bare numbers in cells
      M  | `Temperature = 0.15` |          mode in the header, a whole
                                            assignment inside each cell
    """
    out, labels = {}, {}
    rows, headers = [], None
    for line in list(block_lines) + ["<<flush>>"]:
        # The |---|---| separator is part of the table, not a break in it.
        # Treating it as a break threw the header row away and promoted the
        # first data row to be the header, which turned "1.0" and "0.7" into
        # column names and lost the Thinking/Instruct split entirely.
        if RULE.match(line):
            continue
        cells = _row_cells(line)
        if cells is None:
            if headers and rows:
                _absorb_table(headers, rows, out, labels)
            rows, headers = [], None
            continue
        if headers is None:
            headers = cells
        else:
            rows.append(cells)
    return out, labels


def _absorb_table(headers, rows, out, labels):
    layout_p = any(
        canon(r[0]) and len(r) > 1 and re.match(r"^\s*[0-9]", r[1] or "")
        for r in rows)

    if layout_p:
        for row in rows:
            key = canon(row[0])
            if not key:
                continue
            for label, cell in zip(headers[1:], row[1:]):
                m = re.match(r"^\s*([0-9][0-9,._]*)", cell or "")
                if not m:
                    continue
                value = to_number(m.group(1), key)
                if value is None:
                    continue
                mk = mode_key(label)
                labels.setdefault(mk, clean(label) or "Default")
                out.setdefault(mk, {})[key] = value
        return

    for row in rows:
        for label, cell in zip(headers, row):
            for name, raw in ASSIGN.findall(cell or ""):
                key = canon(name)
                value = to_number(raw, key)
                if not key or value is None:
                    continue
                mk = mode_key(label)
                labels.setdefault(mk, clean(label) or "Default")
                out.setdefault(mk, {}).setdefault(key, value)


BULLET = re.compile(r"^\s*[*\-+]\s+(.*)$")


def parse_bullets(block_lines):
    """Lines like:
        * Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`
        * `temperature = 1.0`
    Bullets are cleaner than the tables (no HTML, no escaped underscores), so
    where both exist and disagree, these win.
    """
    out, labels = {}, {}
    for raw in block_lines:
        m = BULLET.match(raw)
        if not m:
            continue
        body = m.group(1)
        found = ASSIGN.findall(body)
        if not found:
            continue
        head = body.split(":", 1)[0] if ":" in body else ""
        head_clean = clean(head)
        # a label only counts if it is prose, not itself an assignment
        if not head_clean or ASSIGN.search(head) or len(head_clean) > 48:
            mk, label = "default", "Default"
        else:
            mk, label = mode_key(head_clean), head_clean
        labels.setdefault(mk, label or "Default")
        for name, value_raw in found:
            key = canon(name)
            value = to_number(value_raw, key)
            if key and value is not None:
                out.setdefault(mk, {})[key] = value
    return out, labels


def parse_context(block_lines, page_lines):
    """Context hides in prose, not in the tables:
        * **Maximum context window:** `262,144`
        The maximum context length Ministral 3 can reach is `262,144`
    """
    for source in (block_lines, page_lines):
        for raw in source:
            text = clean(raw)
            m = CONTEXT_PHRASE.search(text)
            if not m:
                continue
            found = []
            for num, suffix in NUMBER.findall(text[m.end():]):
                value = to_number(num, "context_length")
                if value is None:
                    continue
                if suffix.upper() == "K":
                    value *= 1024
                elif suffix.upper() == "M":
                    value *= 1024 * 1024
                if 1024 <= value <= 16 * 1024 * 1024:
                    found.append(int(value))
            # Gemma 4 quotes two on one line - 128K for the tiny variants and
            # 262,144 for the rest. Take the larger; it gets clamped anyway.
            if found:
                return max(found)
    return None


# --------------------------------------------------------------------------
# llama.cpp extras
# --------------------------------------------------------------------------

LIFT_FLAGS = ("--jinja", "--flash-attn", "-fa")
SUGGEST_ONLY = ("-ot", "--override-tensor", "--n-cpu-moe", "--cpu-moe")


def parse_extras(page_lines):
    """Pull a couple of safe flags out of the llama.cpp tutorial block.

    Deliberately tiny. -ot ".ffn_.*_exps.=CPU" appears in nearly every
    tutorial and forces expert tensors onto the CPU: right for a 480B MoE
    that does not fit, ruinous for a 27B dense model that does. It is
    reported as a suggestion and never applied.
    """
    lift, suggest = [], []
    for text in page_lines:
        for flag in LIFT_FLAGS:
            if re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(flag), text):
                canonical = "--flash-attn" if flag == "-fa" else flag
                if canonical not in lift:
                    lift.append(canonical)
        for flag in SUGGEST_ONLY:
            m = re.search(r"(?<![\w-])%s\s+(\S+)" % re.escape(flag), text)
            if m:
                hint = "%s %s" % (flag, m.group(1))
                if hint not in suggest:
                    suggest.append(hint)
    return lift, suggest[:2]


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------

@dataclass
class Result:
    settings: dict = field(default_factory=dict)
    modes: tuple = ()
    mode_labels: dict = field(default_factory=dict)
    mode: str = ""
    page: str = ""
    url: str = ""
    section: str = ""
    line: int = 0
    age: str = ""
    error: str = ""
    notes: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    candidates: list = field(default_factory=list)

    def source_line(self):
        """One line for the generated script's header comment."""
        if not self.page:
            return ""
        bits = ['unsloth.ai docs "%s"' % self.page]
        if self.section:
            bits.append(self.section)
        if self.mode:
            bits.append(self.mode_labels.get(self.mode, self.mode))
        return " > ".join(bits)

    def describe(self):
        if not self.settings:
            return self.error or "Nothing found."
        line = self.source_line()
        tail = []
        if self.line:
            tail.append("line %d" % self.line)
        if self.age:
            tail.append(self.age)
        if tail:
            line += "   (%s)" % ", ".join(tail)
        if self.notes:
            line += "\n" + "\n".join(self.notes)
        return line


def recommend(model, cfg=None, mode=None, force_refresh=False):
    """Find the recommended settings for one model. Never raises.

    Resolved once, then saved. A repeat click - including switching between
    Thinking and Instruct - is answered from disk with no network at all.
    """
    cfg = cfg or {}
    family = model_family(getattr(model, "id", ""))

    if not force_refresh:
        saved = load_resolved().get(family)
        if saved:
            built = _result_from(saved, model, mode)
            if built is not None:
                return built

    result, payload = _resolve(model, cfg, mode, force_refresh)
    if payload:
        save_resolved(family, payload)
    return result


def _result_from(saved, model, mode):
    """Rebuild a Result from the saved answer. Returns None if the saved
    shape is from an older version and cannot be trusted."""
    try:
        merged = {k: dict(v) for k, v in saved["merged"].items()}
        modes = tuple(saved["modes"])
        if not merged or not modes:
            return None
    except (KeyError, TypeError, AttributeError):
        return None

    chosen = _pick_mode(modes, model, mode)
    settings = dict(merged.get(chosen) or {})
    if not settings:
        return None
    for key, value in (saved.get("always") or {}).items():
        settings.setdefault(key, value)

    when = saved.get("saved")
    age = "saved %s" % _age_text(time.time() - when) if when else "saved"
    return Result(
        settings={k: (v if isinstance(v, str) else fmt(v))
                  for k, v in settings.items()},
        modes=modes, mode_labels=dict(saved.get("labels") or {}), mode=chosen,
        page=saved.get("page", ""), url=saved.get("url", ""),
        section=saved.get("section", ""), line=int(saved.get("line") or 0),
        age=age, notes=list(saved.get("notes") or []),
        suggestions=list(saved.get("suggestions") or []))


def _resolve(model, cfg, mode, force_refresh):
    """Do the actual lookup. Returns (Result, payload-to-save-or-None)."""
    def fail(**kw):
        return Result(**kw), None

    family = model_family(getattr(model, "id", ""))
    manual_title = (cfg.get("manual_page_map") or {}).get(family, "")

    index, age, err = fetch_index(force=force_refresh)
    if not index:
        return fail(age=age, error=(
            "Could not reach unsloth.ai (%s) and no documentation is cached. "
            "Type the settings in by hand, or try again when you are online."
            % err) if err else "No documentation available.")

    entry, how = find_page(index, family, manual_title)
    titles = sorted(e["title"] for e in index.values())
    if entry is None:
        return fail(age=age, candidates=titles, error=(
            'No Unsloth docs page for "%s", so nothing was filled in. '
            "Pick a page below if you know which one applies." % family))

    text, page_age, page_err = fetch_page(entry, force=force_refresh)
    if not text:
        return fail(age=age, candidates=titles, error=(
            'Found "%s" in the index but could not download it (%s).'
            % (entry["title"], page_err)))

    lines = text.split("\n")
    notes = []
    if how == "older":
        notes.append(
            'No page for "%s" yet, so these are %s settings - the closest '
            "earlier release. Check them before relying on them."
            % (family, entry["title"]))
    elif how == "manual":
        notes.append("Using the page you picked for this model.")

    found = find_settings_block(lines)
    if not found:
        return fail(age=page_age, page=entry["title"], url=entry["url"],
                      candidates=titles, notes=notes,
                      error='Opened "%s" but found no recommended-settings '
                            "section in it." % entry["title"])

    start, end, heading = found
    variant = pick_variant(find_variants(lines, start, end), model)
    if variant:
        block_start, block_end = variant["line"], variant["end"]
        section = "%s > %s" % (heading, variant["title"])
    else:
        block_start, block_end = start, end
        section = heading
    block = lines[block_start:block_end]

    table_vals, table_labels = parse_tables(block)
    bullet_vals, bullet_labels = parse_bullets(block)
    labels = dict(table_labels)
    labels.update(bullet_labels)
    merged = {}
    for source in (table_vals, bullet_vals):        # bullets last: they win
        for mk, values in source.items():
            merged.setdefault(mk, {}).update(values)

    if not merged:
        return fail(age=page_age, page=entry["title"], url=entry["url"],
                      section=section, line=block_start + 1, notes=notes,
                      candidates=titles,
                      error="Could not read any numbers out of that section.")

    modes = tuple(merged.keys())
    chosen = _pick_mode(modes, model, mode)
    settings = dict(merged[chosen])

    doc_ctx = parse_context(block, lines)
    if doc_ctx:
        ctx = min(doc_ctx, getattr(model, "context_max", 0) or doc_ctx)
        if ctx > SAFE_CONTEXT:
            notes.append(
                "Docs list a %s context. Filled in %s instead - the full "
                "window will not fit in 16 GB. Raise it if you have the room."
                % (format(doc_ctx, ","), format(SAFE_CONTEXT, ",")))
            ctx = SAFE_CONTEXT
        settings["context_length"] = ctx

    lift, suggest = parse_extras(lines)
    if lift:
        settings["extra_flags"] = " ".join(lift)

    out = {k: (v if isinstance(v, str) else fmt(v)) for k, v in settings.items()}
    suggestions = ['docs mention %s - only helps if the model does not fit '
                   "in VRAM, so it is not applied" % s for s in suggest]

    # Everything needed to answer this model again without the network,
    # including the other modes so flipping the dropdown stays instant.
    always = {}
    if "context_length" in settings:
        always["context_length"] = settings["context_length"]
    if "extra_flags" in settings:
        always["extra_flags"] = settings["extra_flags"]
    payload = {
        "merged": {mk: dict(vals) for mk, vals in merged.items()},
        "modes": list(modes), "labels": labels, "always": always,
        "page": entry["title"], "url": entry["url"], "section": section,
        "line": block_start + 1, "notes": notes, "suggestions": suggestions,
        "family": family,
    }

    result = Result(
        settings=out, modes=modes, mode_labels=labels, mode=chosen,
        page=entry["title"], url=entry["url"], section=section,
        line=block_start + 1, age=page_age, notes=notes,
        suggestions=suggestions)
    return result, payload


def _pick_mode(modes, model, override=None):
    if override and override in modes:
        return override
    name = str(getattr(model, "id", "")).lower()
    caps = tuple(getattr(model, "capabilities", ()) or ())
    if "instruct" in name and "instruct" in modes:
        return "instruct"
    if ("reason" in name or "think" in name) and "thinking" in modes:
        return "thinking"
    if "coder" in name and "coder" in modes:
        return "coder"
    if "thinking" in caps and "thinking" in modes:
        return "thinking"
    # Capabilities come from /api/tags, which is unavailable whenever the
    # Ollama server is not running yet - and choosing a mode happens before
    # anything is started, so this is the normal case rather than the edge
    # case. Doc order is the honest fallback: these pages lead with the
    # model's default mode.
    if "default" in modes:
        return "default"
    return modes[0]


def remember_page(cfg, model, page_title):
    """Persist a manual page choice so the user only makes it once."""
    family = model_family(getattr(model, "id", ""))
    cfg.setdefault("manual_page_map", {})[family] = page_title
    return cfg


def page_titles(force=False):
    index, _age, _err = fetch_index(force=force)
    return sorted(e["title"] for e in index.values())


# --------------------------------------------------------------------------
# the saved answer
# --------------------------------------------------------------------------

RESOLVED_PATH = os.path.join(state.CACHE_DIR, "resolved.json")


def load_resolved():
    return state.read_json(RESOLVED_PATH, {})


def save_resolved(family, payload):
    """Work this out once, then keep it.

    The index and page caches expire after a day so the docs can be picked
    up when they change. This file does not expire: it is the finished
    answer for a model, and once a model's recommended settings have been
    read successfully there is no reason to go back to the network for them
    - including on a machine that is offline from then on. Refresh re-reads
    it deliberately.
    """
    data = load_resolved()
    payload = dict(payload)
    payload["saved"] = time.time()
    data[family] = payload
    state.write_json(RESOLVED_PATH, data)
    return payload


def forget_resolved(family=None):
    if family is None:
        state.write_json(RESOLVED_PATH, {})
        return
    data = load_resolved()
    data.pop(family, None)
    state.write_json(RESOLVED_PATH, data)
