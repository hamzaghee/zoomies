# Preset research — 3 uncensored Qwen3.8-27B repos

Reference notes for building Zoomies presets. Researched 2026-10-06, updated with the
downloaded files and with live measurements off the running server.

Cards cached in `%LOCALAPPDATA%\Zoomies\cache\cards\` via `optimizer.fetch_card()`:

    JonathanColetti__Qwen3.8-27B-Uncensored-GGUF.md                                     (A)
    huihui-ai__Huihui-Qwen3.8-27B-abliterated-GGUF.md                                   (B)
    DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-...-GGUF.md  (C)
    Qwen__Qwen3.8-27B.md                                                                (base)

## Files picked

| | file | GB | blocks | MTP |
|---|---|---|---|---|
| A | `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` | 16.55 | 64 | **no** |
| A draft | `Qwen3.8-27B-Uncensored-draft-Q8_0.gguf` | 3.16 | 1 (`blk.64` only) | head itself |
| B | `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` | 17.38 | 65 | **yes** (`nextn_predict_layers=1`) |
| C | `Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q5_K_M.gguf` | 21.18 | 65 | **yes** (MTP block 0.45 GB) |

All verified from the headers with `gguf.read()`. All three templates carry
`reasoning_effort` with xhigh/medium/low, so `reasoning_style: template` is valid
everywhere. The draft head downloaded 2026-10-06 and matches the published size byte for
byte; its `general.name` is `Abl 27b`, same as A's, so it is the right lineage.

---

## Measured on this machine — this drives everything below

B is **running right now** at `-c 131072`, f16 KV, and it works. So 131072 is real and
the 64k figures in `presets.json` are conservative. My earlier "draft head drops you to
48k" was extrapolated from those anchors and was wrong — see the real budget below.

Live VRAM with B loaded at 131072 (perf counters, both Radeon RX 6800 XT):

    card 1 (LUID 00011B37)   14.50 GB used   <- measured ceiling is 14.51 GB. Full.
    card 2 (LUID 038EFDFD)   13.41 GB used   <- clean figure 15.65 GB
    total                    27.91 GB

Budget ≈ 30.16 GB (14.51 + 15.65), but card 2's number is the optimistic sticker figure,
not measured, and card 1 is already at the wall. Spare is ~2.2 GB and it is all on card 2.

### KV cost, computed from the header

`full_attention_interval = 4`, so only **16 of 64** layers are full attention. Those have
`head_count_kv 4`, `key_length`/`value_length` 256:

    4 heads x 256 x 2 bytes x 2 (K+V) = 4 KiB per layer per token
    x 16 full-attention layers        = 64 KiB per token

That matches the Swift preset note exactly. So **f16 KV at 131072 = 8.59 GB**;
q8_0 KV ≈ 4.30 GB. The other 48 layers are SSM — constant state, does not grow
with context (~0.2 GB).

### Where the 27.91 GB goes

    weights (17.38 on disk - 0.35 skipped blk.64)   17.03 GB
    KV f16 at 131072                                 8.59 GB
    compute buffer + SSM state + overhead            2.29 GB
                                                    --------
                                                    27.91 GB

**That 2.29 GB is the headroom hiding in plain sight.** The live run passes no `-fa`,
no `-ub`, no `-ngl` and no spec flags. The existing preset notes project the compute
buffer at ~1.60 GB at 64k with `-fa off` (scaling linearly, so ~3.2 GB at 131k) against
~0.10 GB with `-fa on`. Adding `-fa on` should free roughly 2 GB — which is most of what
speculative decoding needs.

No card mentions KV cache or flash attention. I grepped all three for
kv/ctk/ctv/cache-type/flash — nothing. So `-fa on`, `-ub 256` and `kv_cache` stay
Zoomies-side engineering choices, not documented settings. (DavidAU's warning about
aggressive caching is about *prompt* caching and tool calls, unrelated.)

---

## Speculative decoding: yes, and here is what it costs

### B — free, and currently switched off

The MTP head is already in the file, and the log shows it being **discarded**:

    model has unused tensor blk.64.nextn.eh_proj.weight ... ignoring   (x15 tensors, 0.35 GB)

Those tensors do nothing unless `--spec-type draft-mtp` is passed. Turning it on costs
~0.35 GB of weights plus a small draft context — no second file, no download.
`--spec-draft-n-max 2` (fallback; this card publishes no sweep).

At 131072 with `-fa on`: 17.38 + 8.59 + ~0.2 = **26.2 GB** of 30.16. Fits with room.

### A — needs a download, and it is the one with real cost

A is the `noMTP` build, so `--spec-type draft-mtp` cannot work alone. The card documents
the split form:

    llama-server -m Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf \
      --spec-type draft-mtp \
      --model-draft Qwen3.8-27B-Uncensored-draft-Q8_0.gguf \
      -ngl 99 -c 8192

`draft-Q8_0` (3.16 GB) is **now downloaded** — the card's default and the only variant it
measured. (`draft-Q4_0` 1.68 GB exists but its acceptance rate is unmeasured.)

One extra cost the card does not mention: the draft gets **its own KV cache**. Its single
block is full attention, so ~4 KiB/token — about 0.54 GB at 131072, less if
`--ctx-size-draft` is set lower.

At 131072 with `-fa on` and f16 KV: 16.55 + 3.16 + 8.59 + 0.54 + 0.2 = **29.0 GB** of
30.16. Too tight to trust, given card 1 is already at its measured ceiling. Two ways out:

- **q8_0 KV at 131072:** 19.71 + 4.30 + 0.27 + 0.2 = **24.5 GB** — comfortable.
- **f16 KV at 98304:** 19.71 + 6.44 + 0.40 + 0.2 = **26.8 GB** — workable.

So: **context barely moves.** You were right. The 3.16 GB of weights is about 50k tokens
of f16 KV if you paid for it out of context — but you don't have to, because `-fa on`
frees ~2 GB and q8_0 KV frees another ~4.3 GB.

Caveat worth measuring: nothing documents how quantized KV interacts with MTP acceptance.
If acceptance looks poor at q8_0, try f16 at 98304 instead.

**n_max for A's split setup is 2, not 1.** The 1.19–1.28x sweep favouring 1 was run on
the *fused* files. The card's only split measurement is `noMTP-IQ2_M` + `draft-Q8_0` at
n_max 2 (1.30x prose, beating the fused file). Nothing measures `noMTP-Q4_K_M` + draft,
so sweep 1 vs 2 once it is running.

Also: the draft head was trained against the unmodified model, so acceptance may fall
slightly. Output quality is unaffected — every token is verified against the target.

### Decision

| | spec decode | cost | n_max |
|---|---|---|---|
| A | yes, `draft-Q8_0` downloaded | 3.16 GB + 0.54 GB draft KV; pay with q8_0 KV | 2, then sweep 1 vs 2 |
| B | yes, free — head already in the file | 0.35 GB | 2 |
| C | yes, it is an MTP file | 0.45 GB, already in the 21.18 GB | 2 (card's predict-2 reference) |

All three get it. A is the only one needing a second file.

---

## Base fallback (Qwen/Qwen3.8-27B)

    Thinking:      temp 1.0  top_p 0.95  top_k 20  min_p 0.0  presence 0.0  repeat 1.0
    Non-thinking:  temp 0.7  top_p 0.80  top_k 20  min_p 0.0  presence 1.5  repeat 1.0

Thinking on by default. `reasoning_effort`: xhigh (default), medium, low. Context 262,144.
YaRN is vLLM/SGLang/TokenSpeed only — not reachable from llama.cpp.

Sampling documented per repo, via `optimizer.find_recommendation()`:
A **none**, B **none**, C **yes (its own set)**. All three declare
`base_model: Qwen/Qwen3.8-27B`, so the fallback is legitimate.

---

## A — `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` (16.55 GB)

Heretic abliteration at bf16, LoRA merged into the bf16 base — not a quantized
round-trip. Refusals 12/100 vs base 98/100. Capability −0.5 mean, within noise.

Quant notes for Q4_K_M:
- Fused Q4_K_M PPL 7.1814 vs f16 7.1557 — inside the noise band, and the card says the
  `noMTP` twins "measure identically". Quality is fine.
- But the card asks you to evaluate *refusal* behaviour on **Q6_K or Q8_0**, because that
  is its least stable property and low bits compound it. Q4_K_M is below that tier and is
  not separately characterised.
- The 12/100 figure was **measured with thinking closed**, so it does not describe the
  model at `xhigh`. Card: may differ "in either direction".

### Settings

    context_length   131072  with q8_0 KV; use 98304 if staying on f16
    gpu_layers       99      card
    parallel         1       fallback
    kv_cache         q8_0    needed to fit the draft head at 131072; legal with -fa on
    sampling = base thinking set (all FALLBACK — card documents none)
    reasoning        xhigh   fallback; see thinking-mode caveat
    reasoning_style  template
    extra_flags      --no-mmproj --jinja -fa on -ub 256
                     --spec-type draft-mtp --spec-draft-n-max 2
                     --model-draft C:\Users\Hamza\.cache\huggingface\hub\Qwen3.8-27B-Uncensored-draft-Q8_0.gguf

Needs llama.cpp PR #22673 for MTP — already verified on this machine (build 10909).
`--no-mmproj`: repo ships `mmproj-*`, auto-discovered by prefix, not downloaded.
`kv_cache q8_0` is a Zoomies-side choice — no card discusses KV. If MTP acceptance looks
poor, fall back to f16 KV at 98304.

---

## B — `Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf` (17.38 GB)

The **UD series**, from `unsloth/Qwen3.8-27B-GGUF`. Ablated layers **17–52**; the first
15 are left alone, which the card says "helps retain more of the original model's
performance".

Dodges all three hazards in that card:
- **Not Ternary/Bonsai** → no PrismML fork, stock llama.cpp runs it.
- **Not a `_K_L` file** → the non-standard upcast naming doesn't apply.
- **MTP head confirmed** (65 blocks, `nextn_predict_layers=1`), matching "MTP and visual
  has not been modified".

Documents no sampling, no chat template, no benchmarks, no refusal rate, no PPL.
Abliteration via `remove-refusals-with-transformers`, which the card itself calls "a
crude, proof-of-concept implementation". Research/controlled use only, per its warnings.

Note the live run is an ad-hoc launch, not a preset: Instruct-mode sampling
(temp 0.7 / top_p 0.8 / presence 1.5), `enable_thinking:false`, `--device Vulkan0,Vulkan1`
pinned. The preset should use the thinking set and leave `--device` blank — Vulkan indices
reshuffle on reboot.

### Settings

    context_length   131072  measured working today
    gpu_layers       99      FALLBACK — card gives no -ngl
    parallel         1       FALLBACK
    kv_cache         f16     start here; q8_0 if it spills
    sampling = base thinking set (all FALLBACK — card documents none)
    reasoning        xhigh   FALLBACK
    reasoning_style  template
    extra_flags      --no-mmproj --jinja -fa on -ub 256
                     --spec-type draft-mtp --spec-draft-n-max 2

Card's only server guidance is `llama-cli -m ... -c 262144`, latest llama.cpp.
262144 is the trained max, not a budget this machine can hold.

---

## C — `...NEO-CODER-MAX-MTP-Q5_K_M.gguf` (21.18 GB)

Best-documented of the three. Multi-stage fine-tune + merge, then Heretic'd again (ARA).
NEO imatrix, output tensor at full 16-bit in all quants, MTP tensors pinned to Q8_0.
256k context. "TURBO" = thinking tokens cut to 1/2–1/10 (median ~2/3).
Header confirms 65 blocks, `nextn_predict_layers=1`, MTP block 0.45 GB.
Internal name is `Qwen3.8 27B Brainwaves NM HERETIC BR LOA1`.

**Q5_K_M lands in the card's "better" tier for tool calling** — it asks for Q4_K_M
minimum, says "q5ks/5km better", and recommends Q6. Good pick for agent work.

**But it is the largest file of the three at 21.18 GB, so context is the constraint.**
At 131072 with f16 KV: 21.18 + 8.59 + 0.2 = 29.97 GB against a 30.16 GB budget — no
margin at all. Options:

| KV | context | total | verdict |
|---|---|---|---|
| q8_0 | 131072 | 25.7 GB | comfortable |
| f16 | 81920 | 26.8 GB | workable |
| f16 | 65536 | 25.7 GB | safe |
| f16 | 131072 | 30.0 GB | no |

**Its own sampling (not fallback):**

    Thinking, general:        temp 1.0  top_p 0.95  top_k 20  min_p 0.0  presence 0.0  repeat 1.0
    Thinking, precise CODING: temp 0.6  top_p 0.95  top_k 20  min_p 0.0  presence 0.0  repeat 1.0
    Non-thinking:             temp 0.7  top_p 0.80  top_k 20  min_p 0.0  presence 1.5  repeat 1.0

→ For a coding preset, **temp 0.6 is documented**. Only place across all three repos where
a model's own card beats the base fallback on sampling.

**Tool calling:** min Q4_K_M, Q5_K_S/M better, **Q6 recommended**; temp .6/.7;
rep pen 1 (off). Below Q4_K_M "may have issues". Aggressive caching may impair calls.

**MTP rules:** temp ≤ 1 (higher degrades MTP) · rep pen 1 (off) · if acceptance < 50% at
predict-2, switch to the regular non-MTP quants. So **n_max 2** is its reference point.
Q4_K_S regular ~75 t/s vs MTP ~90+ t/s at 60% acceptance (5090, LMStudio).
For creative work or temp > 1, use the regular quants.

**Reasoning:** xhigh/medium/low. Via jinja at the very top:
`{%- set reasoning_effort = 'medium' %}`. Note **medium turns off system-prompt injection**
entirely. Card warns the reasoning change is "a major change — carefully test it".

**Chat/roleplay only:** smoothing_factor 1.5, or rep pen 1.1–1.15. Not for coding —
contradicts the MTP rep-pen-1 rule.

**Files** — MTP twins ~0.45 GB larger; `LOW-MTP` reduces footprint:

| file | GB | | file | GB |
|---|---|---|---|---|
| LOW-MTP-IQ4_XS | 15.31 | | MTP-Q5_K_S | 20.63 |
| MTP-IQ2_M | 12.12 | | MTP-Q5_K_M | 21.18 |
| MTP-IQ3_M | 14.53 | | LOW-MTP-Q6_K | 22.43 |
| MTP-IQ4_XS | 17.03 | | MTP-Q6_K | 24.03 |
| MTP-Q4_K_S | 17.54 | | MTP-Q8_0 | 30.24 |
| MTP-Q4_K_M | 18.50 | | mmproj F16/BF16 | 0.93 |

Benchmarks (Nightmedia, instruct mode): ARC-C 0.735 mxfp8 / 0.719 mxfp4 vs base
Qwen3.8-27B 0.591 / 0.581. De-censoring: stage 1 refusals 0/100 (KLD 0.0535), stage 2
11/100 (KLD 0.0025).

### Settings

    context_length   131072  with q8_0 KV; 81920 if staying on f16
    gpu_layers       99      fallback
    parallel         1       fallback
    kv_cache         q8_0    needed at 131072 — 21.18 GB of weights leaves no room on f16
    temperature      0.6     CARD — precise coding set
    top_p            0.95    CARD
    top_k            20      CARD
    min_p            0.0     CARD
    presence_penalty 0.0     CARD
    repeat_penalty   1.0     CARD — and required by the MTP rule
    reasoning        xhigh   card default; TURBO already cuts the tokens
    reasoning_style  template
    extra_flags      --no-mmproj --jinja -fa on -ub 256
                     --spec-type draft-mtp --spec-draft-n-max 2   (MTP files only)

---

## Fit arithmetic (budget ≈ 30.16 GB, KV f16 = 64 KiB/token)

| setup | weights | KV @131072 | total | verdict |
|---|---|---|---|---|
| B today (no -fa, no MTP) | 17.03 | 8.59 f16 | 27.91 measured | at the wall |
| B + MTP, -fa on | 17.38 | 8.59 f16 | ~26.2 | fits |
| A + draft, -fa on | 19.71 | 8.59 f16 + 0.54 draft | ~29.0 | too tight |
| A + draft, q8_0 KV | 19.71 | 4.30 + 0.27 draft | ~24.5 | comfortable |
| C + MTP, -fa on | 21.18 | 8.59 f16 | ~30.0 | no |
| C + MTP, q8_0 KV | 21.18 | 4.30 | ~25.7 | comfortable |

So B runs on f16 KV; A and C want q8_0 KV to hold 131072. All three fit with MTP on.

Card 1's ceiling (14.51 GB) is measured; card 2's (15.65 GB) is the sticker figure and
those keys orphan on reboot. Load once and let the spill alert set the real number.

---

## Open items

All three files are downloaded and verified. Ready to write presets.

1. **A:** `xhigh`, or thinking off to match the mode its refusal number was measured in?
   Docs don't settle it — the only question I still need answered.
2. Confirm q8_0 KV doesn't hurt MTP acceptance on A and C. Nothing documents the
   interaction. Fallback is f16 KV at 98304 (A) / 81920 (C).
3. Sweep A's n_max 1 vs 2 — the sweep favouring 1 was fused-only, the split measurement
   used 2.
4. Sweep `-fa on` vs off and `-ub` at 131072; the bench found they interact, and the
   existing figures come from an 8k profile.
5. Keep `opencode.jsonc` context limits in step with each preset.
6. Leave `--device` out of all three presets — Vulkan indices reshuffle on reboot.

---

## What was written, 2026-10-06

### Zoomies presets (7)

All three files get a thinking and a non-thinking preset. C gets a third because its
card separates general thinking (temp 1.0) from precise coding thinking (temp 0.6).

| model | preset | reasoning | temp | use_for |
|---|---|---|---|---|
| A | Coding agent, thinking, 128k | xhigh | 1.0 | agent |
| A | Instruct, non-thinking, 128k | off | 0.7 | chat |
| B | Coding agent, thinking, 128k | xhigh | 1.0 | agent |
| B | Instruct, non-thinking, 128k | off | 0.7 | chat |
| C | Coding agent, thinking, 128k | xhigh | **0.6** | agent |
| C | General chat, thinking, 128k | xhigh | 1.0 | chat |
| C | Instruct, non-thinking, 128k | off | 0.7 | chat |

All at 131072 context, q8_0 KV, gpu_layers 99, parallel 1, and
`--no-mmproj --jinja -fa on -ub 256 --spec-type draft-mtp --spec-draft-n-max 2`;
A adds `--model-draft <hub>\Qwen3.8-27B-Uncensored-draft-Q8_0.gguf`.

`reasoning.detect()` confirms all three templates offer off/low/medium/xhigh with
default xhigh, so `reasoning_style: template` is right and the levels are real.

Non-thinking is not just a switch: the base card gives Instruct mode its own sampling
(temp 0.7 / top_p 0.80 / presence 1.5), so flipping thinking off without changing those
would leave presence_penalty at 0 where the card asks for 1.5.

One bug caught in review: the `--model-draft` path was first written quoted, and
`shlex.split(posix=False)` keeps the quotes in the token, which would have reached
llama.cpp as part of the filename. The path has no spaces, so the quotes came out.

### Harness configs

Reconciled every .gguf on disk, not just the three new ones. The draft head is
deliberately not listed anywhere as a model — it is a companion, not a servable model.

| | before | after |
|---|---|---|
| opencode.jsonc | 11 entries, 8 models missing | 19 entries, all 17 on disk present |
| dsh settings.yaml | 7 entries, 10 models missing | 17 entries, all present |
| hermes config.yaml | pinned to a model that was not loaded | repointed to the loaded one |
| openclaw | **not set up — nothing to configure** | unchanged |

opencode entries are sourced per model: sampling and context from that model's Zoomies
preset, variants from `reasoning.detect()` against its own template, output from
`backends.max_tokens_for()`. Three different reasoning switches turned up:

- Qwen3.8 family — `enable_thinking` + `reasoning_effort`, off/low/medium/xhigh
- Ornith — `enable_thinking` only, off/on
- Muse Glimmer — `reasoning_strength`, low/medium/high/xhigh, **no off level**
- gpt-oss — `reasoning_effort` low/medium/high, no off; `detect()` cannot enumerate
  these (the template reads the variable without listing values), so they come from
  OpenAI's own card

### Left alone, deliberately

- **`Ornith-1.5-35B-Q4_K_M` has no Zoomies preset.** Its opencode and dsh entries borrow
  sampling and 64k context from the old `ornith-1.5:35b` Ollama entry, which is flagged in
  both files. Writing it a preset needs its own docs read first.
- **Two opencode entries point at files not on disk** — `Qwen3.8-27B-UD-Q6_K_M` and
  `unsloth/GLM-4.7-Flash-GGUF:UD-Q4_K_XL`. Removing config is destructive and they may be
  re-downloaded, so they stay until you say otherwise.
- **hermes has no sync module.** opencode.py and dsh.py both rewrite their configs on a
  model load; nothing does this for hermes, so its single pin goes stale on every swap.
- **hermes `context_length_cache.yaml`** still lists four models on the retired 8888
  Unsloth endpoint. It is a cache Hermes writes itself, so it was not hand-edited.

### Backups

    presets.json.bak-20261006-205046-pre-uncensored-trio
    opencode.jsonc.bak-20261006-205749-pre-uncensored-trio
    settings.yaml.bak-20261006-205749-pre-uncensored-trio      (dsh)
    config.yaml.bak-20261006-205749-pre-uncensored-trio        (hermes)

---

## Follow-ups, 2026-10-06 (later)

### Ornith 1.5 35B — corrected from its card

`Ornith-1.5-35B-Q4_K_M.gguf` (21.71 GB) **is** the A3B the card describes — the header
confirms arch `qwen35moe`, 256 experts with 8 used per token, embedding length 2048,
41 blocks, 262,144 trained context. So the card applies first-hand, not by borrowing.

Documented: *"Recommended sampling parameters > For general tasks: temperature=0.6,
top_p=0.95, top_k=20."* It also says temperature=1.0 reproduces its published benchmarks —
a reproduction setting, not advice, so it is not used. Nothing is given for min_p,
presence or repeat, so those stay blank. It is a reasoning model: the assistant turn opens
with a `<think>` block by default, and its template offers only off/on — no effort levels.

**KV is unusually cheap on this architecture.** `full_attention_interval 4` over 40 real
layers leaves only 10 full-attention layers, each with just 2 KV heads at key/value
length 256:

    2 heads x 256 x 2 bytes x 2 (K+V) = 2 KiB per layer per token
    x 10 full-attention layers        = 20 KiB per token

That is under a third of the dense Qwen3.8-27B's 64 KiB/token. f16 KV at 131072 is only
2.68 GB, so 21.71 + 2.68 + ~0.2 ≈ **24.6 GB** — comfortable. The full 262,144 projects to
~27.3 GB and looks reachable, but is not the default because card 1's 14.51 GB ceiling
binds first.

So the 65536 borrowed from the retired `ornith-1.5:35b` Ollama entry was far too
conservative. Raised to 131072 in its new presets and in both configs.

`-ub 1024`, not the usual 256: the old Ollama preset measured 2,151 prompt t/s and
100.2 gen t/s on these same weights at 1024, +8.0% over the best single setting, while the
9B prefers 256 and both gpt-oss files prefer the default 512. Micro-batch is per-model.

Two new presets written: **Coding agent, thinking, 128k** and **Fast, no thinking, 128k**.
Sampling is identical in both, deliberately — unlike the Qwen3.8 cards, this one documents
no separate non-thinking recipe, so inventing one would not be grounded.

All 9 new presets were checked with `backends.reasoning_kwargs()` against each model's real
template: every level resolves, none falls back.

### hermes.py — the sync module

`hermes.py` mirrors `dsh.py`: `config_path()` / `config_stamp()` / `plan_limits()` /
`write()`, line-based edits so the file's comments survive, atomic write via tmp +
`os.replace`. Wired into `app.py` as a third independent pass beside opencode and dsh
(4 changes: import, two state fields, call site, method).

It writes only the three keys it owns — `model.context_length`, `model.default`, and the
`model:` line of each `fallback_providers` entry for the local provider.

Three safety choices, since this file is dense with comments and is not ours:

- **Nothing is ever inserted.** A missing key is reported, not created. dsh inserts a
  missing `contextWindow` because it owns that entry; here an absent key just means Hermes
  falls back to its own default, and guessing where to add a line in someone else's
  commented file is not worth it.
- **Every edit is scoped to a section span.** `model:` at four spaces also appears under
  the voice and reference_models sections (`model: base`, `model: whisper-1`,
  `model: anthropic/...`) and a loose pattern would rewrite those.
- **The result is parsed before it is committed** when PyYAML is importable; a write that
  would leave the file unloadable raises instead. A missing PyYAML only skips the check.

Tested dry-run against the real config:

| case | result |
|---|---|
| same model, same context | 0 edits — idempotent |
| different model + context | 3 correct edits; 68 comments and 331 lines both unchanged; the three unrelated `model:` keys untouched |
| nothing loaded | 0 edits, pin left alone |
| server on a port Hermes does not point at | 0 edits, note explaining why |
| deliberately corrupt edit | `write()` refused, file unchanged |

Reverting it is two steps: delete `hermes.py` and undo the four `app.py` edits (a copy of
the original is in the scratchpad). Nothing else imports it, so removing it cannot affect
opencode or dsh.

### OpenClaw — still cannot be configured

"OpenClaw Companion" is the tray app at
`%LOCALAPPDATA%\OpenClawTray\OpenClaw.Tray.WinUI.exe` (resolved from the desktop
shortcut), and it is installed. But it keeps its model config inside a **WSL2 distro it
has not created**: its own `default-config.json` sets `"DistroName": "OpenClawGateway"`,
and `wsl --list --all` shows only `Ubuntu` and `docker-desktop`. The default Ubuntu distro
has no openclaw binary and no `~/.openclaw` config. `gateways.json` is empty
(`"gateways": [], "activeId": null`).

So the gateway has never been provisioned and there is no live config to edit. Run the
Companion's setup once to create the distro; the config will then be inside it (the
August exports in `~/Downloads` and `~/.ollama/backup/openclaw/` show the shape:
`models.providers.<name>.models[]` with `id`, `contextWindow`, `maxTokens`, `reasoning`),
and it can be filled in from there.
