r"""
Zoomies - how much VRAM a llama.cpp load will take, before loading it.

Worked out from the .gguf header and the launch settings alone, so it is
instant and never touches a GPU. Per card, because layers are split across
cards and one card can overflow while the total fits:

  weights   the offloaded layers' tensors, plus the output layer on the last
            card. The input embeddings stay in system RAM. A multi-token-
            prediction layer (nextn) is never loaded.
  KV cache  context x bytes per token, over the layers that keep one:
              - every 4th layer on Qwen3.5 / 3.6 / 3.8 and Ornith, whose
                other layers are recurrent,
              - only the attention layers on Nemotron,
              - sliding-window layers (Gemma 4, Muse Glimmer, Laguna) hold
                their window, not the whole context.
  state     the fixed recurrent state of those other layers, per slot.
  compute   llama.cpp's scratch space. The one rough part: it depends on
            the batch size, flash attention and the vocabulary, and is
            estimated here rather than read from anywhere.

How llama.cpp places layers is mirrored from llama-model.cpp: split in
proportion to each card's free memory unless --tensor-split says otherwise,
with the output layer counted as one more layer at the end. "Free" is what
the Vulkan driver reports, and on this machine it reports each card as empty:
Qwen3.8 at 64k measured 9.36 / 9.93 GB with 3.7 GB of desktop on card 0,
which is an even split. So the split here follows card size.
"""

import bisect
import math
from dataclasses import dataclass, field

import gguf
import reasoning
import state

GB = 1024 ** 3
MB = 1024 ** 2

# Bytes per value in the KV cache (ggml block sizes: q8_0 packs 32 values in
# 34 bytes, and so on).
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32,
            "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32,
            "q4_0": 18 / 32, "iq4_nl": 18 / 32}

# Laguna does not write its layer pattern into the header; llama.cpp hardcodes
# it (src/models/laguna.cpp: full attention at il % 4 == 0).
SWA_PERIOD = {"laguna": 4}

# Fixed cost per card that no header mentions: the Vulkan context and
# llama.cpp's own small buffers. Set from measured loads (flash attention
# on): Qwen3.8 64k read 0.1 GB per card under this estimate, Devstral Q6 49k
# and Laguna 64k 0.2 GB over it - so expect about +-0.25 GB per card.
CARD_OVERHEAD = 250 * MB

# How much room a card should keep free. Windows spills to system RAM
# silently rather than failing, so "just fits" is not good enough.
DEFAULT_MARGIN = 512 * MB


@dataclass
class Card:
    name: str
    luid: str
    total: int                    # dedicated VRAM
    other: int = 0                # used by everything else
    weights: int = 0
    kv: int = 0
    state: int = 0
    compute: int = 0
    layers: int = 0
    # What this card will actually hand out. Windows keeps a reserve for the
    # desktop, so a 16 GB card can start spilling at 13.6 GB; 0 means
    # nothing has been measured yet and the sticker figure has to do.
    budget: int = 0

    @property
    def model(self):
        return self.weights + self.kv + self.state + self.compute + CARD_OVERHEAD

    @property
    def used(self):
        return self.other + self.model

    @property
    def limit(self):
        return self.budget or self.total

    @property
    def spare(self):
        return self.limit - self.used


@dataclass
class Estimate:
    cards: list = field(default_factory=list)
    ctx: int = 0
    max_ctx: int = 0              # largest context leaving the margin free
    notes: list = field(default_factory=list)
    problem: str = ""
    margin: int = DEFAULT_MARGIN

    @property
    def holds(self):
        """The arithmetic works, with nothing to spare."""
        return not self.problem and all(c.spare >= 0 for c in self.cards)

    @property
    def fits(self):
        """Room for the model and the margin on every card.

        The margin is not politeness. Other apps move around while a model
        loads, and the load itself peaks above what it settles at, so a card
        with nothing spare is a card that spills.
        """
        return not self.problem and all(c.spare >= self.margin
                                        for c in self.cards)

    @property
    def tight(self):
        """Fits on paper, but without the margin somewhere."""
        return self.holds and not self.fits


# --------------------------------------------------------------------------
# which cards
# --------------------------------------------------------------------------

def adapters():
    """Cards in the order llama.cpp's Vulkan backend numbers them.

    Vulkan0, Vulkan1... follow DXGI's order of the non-software adapters,
    integrated graphics included (this machine: RX 6800 XT, UHD 770, RX 6800
    XT, so the two RX cards are Vulkan0 and Vulkan2). That order is an
    assumption - llama.cpp is never asked, because asking starts Vulkan.
    """
    out, seen = [], set()
    for a in state.enumerate_adapters():
        # A driver reinstall leaves a card listed twice under the same
        # hardware key (see state.physical_adapters); count it once.
        if a["is_software"] or a["key"] in seen:
            continue
        seen.add(a["key"])
        out.append(a)
    return out


def cards_for(tokens, found=None):
    """The Cards a launch with these Extra flags will use."""
    found = adapters() if found is None else found
    wanted = reasoning._flag_value(tokens, ("-dev", "--device"))
    if wanted:
        picked = []
        for name in wanted.split(","):
            name = name.strip()
            digits = "".join(ch for ch in name if ch.isdigit())
            if name.lower().startswith("vulkan") and digits and \
                    int(digits) < len(found):
                picked.append(found[int(digits)])
        chosen = picked
    else:
        # What build_launch does when no --device is given: dedicated only.
        chosen = [a for a in found if a["vram"] >= GB]
    limits = state.load_vram_limits()
    return [Card(name=state.display_label(a, chosen), luid=a["luid"],
                 total=int(a["vram"]),
                 budget=state.vram_budget(limits, a["luid"], a["vram"]))
            for a in chosen]


# --------------------------------------------------------------------------
# the model's layers
# --------------------------------------------------------------------------

def _per_layer(value, n):
    if isinstance(value, list):
        return [int(v or 0) for v in value[:n]] + [0] * max(0, n - len(value))
    return [int(value or 0)] * n


class Layers:
    """Per-layer facts from the header."""

    def __init__(self, h):
        self.h = h
        self.arch = h.arch
        total = int(h.get("block_count") or len(h.layer_bytes))
        nextn = int(h.get("nextn_predict_layers") or 0)
        self.n_all = total                # what llama.cpp splits over
        self.n = total - nextn            # layers that run and keep caches
        n = self.n
        self.n_embd = int(h.get("embedding_length") or 0)
        self.n_head = _per_layer(h.get("attention.head_count"), n)
        self.n_head_kv = _per_layer(
            h.get("attention.head_count_kv", h.get("attention.head_count")), n)
        head = self.n_embd // max(1, max(self.n_head) or 1)
        self.k_len = int(h.get("attention.key_length") or head)
        self.v_len = int(h.get("attention.value_length") or head)
        self.k_len_swa = int(h.get("attention.key_length_swa") or self.k_len)
        self.v_len_swa = int(h.get("attention.value_length_swa") or self.v_len)
        self.n_vocab = int(h.get("vocab_size")
                           or h.lengths.get("tokenizer.ggml.tokens") or 0)
        ff = h.get("feed_forward_length")
        self.n_ff = max(ff) if isinstance(ff, list) else int(ff or 0)
        self.n_expert_ff = int(h.get("expert_feed_forward_length") or 0) * \
            int(h.get("expert_used_count") or 0)
        self.window = int(h.get("attention.sliding_window") or 0)
        self.notes = []

        # Which layers keep a KV cache, which a recurrent state.
        interval = int(h.get("full_attention_interval") or 0)
        if interval:                      # Qwen3.5 family: every Nth layer
            self.attn = [(i + 1) % interval == 0 for i in range(n)]
        elif isinstance(h.get("attention.head_count_kv"), list):
            self.attn = [k > 0 for k in self.n_head_kv]   # Nemotron-H
        else:
            self.attn = [True] * n
        self.recurrent = [not a for a in self.attn] \
            if h.get("ssm.inner_size") else [False] * n
        if isinstance(ff, list) and not interval and h.get("ssm.inner_size"):
            # Nemotron-H: layers with neither attention nor a feed-forward
            # block are the Mamba ones; MoE layers hold no state.
            self.recurrent = [not a and not f for a, f in
                              zip(self.attn, _per_layer(ff, n))]

        pattern = h.get("attention.sliding_window_pattern")
        if self.window and isinstance(pattern, list):
            self.swa = [bool(p) for p in pattern[:n]] + [False] * (n - len(pattern))
        elif self.window and self.arch in SWA_PERIOD:
            period = SWA_PERIOD[self.arch]
            self.swa = [i % period != 0 for i in range(n)]
        else:
            self.swa = [False] * n
            if self.window:
                self.notes.append("The header gives a %d-token sliding window "
                                  "but not which layers use it, so every layer "
                                  "is counted with the full context - an upper "
                                  "bound." % self.window)

        # Recurrent state per slot, in f32: the convolution window plus the
        # state matrix (Mamba2 and Qwen's gated delta net share the shape).
        d_inner = int(h.get("ssm.inner_size") or 0)
        d_state = int(h.get("ssm.state_size") or 0)
        groups = int(h.get("ssm.group_count") or 1)
        conv = int(h.get("ssm.conv_kernel") or 4)
        self.state_per_layer = 4 * ((conv - 1) * (d_inner + 2 * groups * d_state)
                                    + d_inner * d_state)

        lb = h.layer_bytes
        self.weight = [int(lb.get(str(i), 0)) for i in range(n)]
        other = {k: v for k, v in h.other_bytes.items()
                 if not k.startswith(("v.", "mm.", "token_embd", "rope_freqs"))}
        # Tied embeddings: no output tensor, so llama.cpp puts a copy of the
        # embedding table on the card as the output layer.
        if "output.weight" not in other:
            other["output.weight"] = h.other_bytes.get("token_embd.weight", 0)
        self.output = sum(other.values())
        vision = [k for k in h.other_bytes if k.startswith(("v.", "mm."))]
        if vision:
            self.notes.append("This file also carries a vision encoder, which "
                              "is not counted.")

    def kv_per_token(self, i, k_bytes, v_bytes):
        if not self.attn[i]:
            return 0.0
        k, v = (self.k_len_swa, self.v_len_swa) if self.swa[i] else \
            (self.k_len, self.v_len)
        return self.n_head_kv[i] * (k * k_bytes + v * v_bytes)


# --------------------------------------------------------------------------
# the estimate
# --------------------------------------------------------------------------

def _int(settings, key, default):
    try:
        return int(float(str(settings.get(key) or "").replace(",", "")))
    except ValueError:
        return default


def _pad(n, to=256):
    return int(math.ceil(n / float(to)) * to)


class Setup:
    """The launch settings that change memory, read the way llama.cpp does."""

    def __init__(self, settings):
        tokens = reasoning.split_flags(settings.get("extra_flags"))
        flag = lambda *names: reasoning._flag_value(tokens, names)
        self.tokens = tokens
        self.ctx = _int(settings, "context_length", 0)
        self.ngl = _int(settings, "gpu_layers", -1)
        self.np = max(1, _int(settings, "parallel", 1))
        self.ub = int(flag("-ub", "--ubatch-size") or 512)
        self.fa = (flag("-fa", "--flash-attn") or "auto").lower() not in ("off", "0")
        kv = str(settings.get("kv_cache") or "").strip() or "f16"
        self.k_type = (flag("-ctk", "--cache-type-k") or kv).lower()
        self.v_type = (flag("-ctv", "--cache-type-v") or kv).lower()
        if not self.fa and self.v_type not in ("f16", "bf16", "f32"):
            self.v_type = "f16"           # what build_launch does, too
        self.kv_on_gpu = not any(t in ("-nkvo", "--no-kv-offload") for t in tokens)
        self.swa_full = "--swa-full" in tokens
        split = flag("-ts", "--tensor-split")
        self.split = [float(x) for x in split.replace("/", ",").split(",")
                      if x.strip()] if split else None
        self.cpu_moe = any(t in ("-ot", "--override-tensor", "--cpu-moe",
                                 "-cmoe", "--n-cpu-moe", "-ncmoe")
                           for t in tokens)
        self.mmproj = "--no-mmproj" not in tokens


def estimate(model, settings, other=None, found=None, margin=DEFAULT_MARGIN):
    """Estimate for a model and the form's settings.

    other: {luid: bytes} used on each card by everything else - the desktop,
    browsers, another model. Missing cards count as empty.
    """
    path = getattr(model, "gguf_path", "") or ""
    if not path:
        return Estimate(problem="No .gguf file to read.")
    try:
        h = gguf.read(path)
    except (OSError, ValueError) as exc:
        return Estimate(problem="Could not read the model file: %s" % exc)
    setup = Setup(settings)
    if setup.ctx <= 0:
        return Estimate(problem="Fill in Context to estimate VRAM.")
    cards = cards_for(setup.tokens, found)
    if not cards:
        return Estimate(problem="No graphics card found.")
    for c in cards:
        c.other = int((other or {}).get(c.luid, 0))
    layers = Layers(h)
    est = _fill(layers, setup, cards, setup.ctx, margin)
    est.notes = list(layers.notes)
    if setup.cpu_moe:
        est.notes.append("Extra flags move tensors to system RAM (-ot / "
                         "--cpu-moe), which this does not model - the real "
                         "use is lower.")
    if setup.mmproj and getattr(model, "source", "") == "hf-cache":
        est.notes.append("A vision projector may load too (no --no-mmproj); "
                         "it is not counted.")
    est.max_ctx = _largest_ctx(layers, setup, cards, margin, h)
    return est


def _fill(layers, setup, cards, ctx, margin):
    """Place the layers on the cards and cost everything for one context."""
    cards = [Card(c.name, c.luid, c.total, c.other, budget=c.budget)
             for c in cards]
    n, n_all = layers.n, layers.n_all
    # llama.cpp offloads the last layers, and counts the output layer (and
    # any unused nextn layer) as positions in the split.
    ngl = n_all + 1 if setup.ngl < 0 else min(setup.ngl, n_all + 1)
    first_gpu = max(n_all + 1 - ngl, 0)

    # Split: --tensor-split, else card size (see the module docstring).
    weights = setup.split if setup.split and len(setup.split) >= len(cards) \
        else [c.total for c in cards]
    weights = weights[:len(cards)]
    total = float(sum(weights)) or 1.0
    bounds, run = [], 0.0
    for w in weights:
        run += w / total
        bounds.append(run)

    def card_for(index):
        """llama.cpp: upper_bound over the cumulative split, over the
        offloaded layers plus the output layer."""
        slot = bisect.bisect_right(bounds, (index - first_gpu) / float(ngl))
        return cards[min(slot, len(cards) - 1)]

    k_bytes = KV_BYTES.get(setup.k_type, 2.0)
    v_bytes = KV_BYTES.get(setup.v_type, 2.0)
    cells = _pad(ctx)
    swa_cells = cells if setup.swa_full else min(
        cells, _pad(layers.window * setup.np + setup.ub))

    for i in range(first_gpu, n):
        card = card_for(i)
        card.layers += 1
        card.weights += layers.weight[i]
        if setup.kv_on_gpu:
            per_token = layers.kv_per_token(i, k_bytes, v_bytes)
            card.kv += int(per_token * (swa_cells if layers.swa[i] else cells))
        if layers.recurrent[i]:
            card.state += layers.state_per_layer * setup.np
    last = None
    if ngl > 0:
        last = card_for(n_all)
        last.weights += layers.output

    # Compute scratch - rough. Activations for a micro-batch on every card
    # holding layers, and without flash attention the attention score matrix
    # for the whole context. The logits are read back into system RAM; with
    # them counted on the last card, Qwen3.8 came out 0.2 GB over measured.
    ub = setup.ub
    act = ub * 4 * (layers.n_embd * 8 + max(layers.n_ff, layers.n_expert_ff) * 2)
    attn = 0
    if not setup.fa:
        attn = ub * cells * max(layers.n_head) * 4 * 2
    for card in cards:
        if card.layers:
            card.compute = act + attn
    return Estimate(cards=cards, ctx=ctx, margin=margin)


def _largest_ctx(layers, setup, cards, margin, h):
    """Largest context (a multiple of 1024, up to what the model was trained
    for) that leaves the margin free on every card."""
    trained = int(h.get("context_length") or 0) or 1 << 20

    def ok(ctx):
        est = _fill(layers, setup, cards, ctx, margin)
        return all(c.spare >= margin for c in est.cards)

    lo, hi = 0, trained // 1024
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid * 1024):
            lo = mid
        else:
            hi = mid - 1
    return lo * 1024


def gb(n):
    return "%.1f" % (n / float(GB))
