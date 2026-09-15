ZOOMIES
An automated LLM optimizer and agnostic launcher.
===================================================================

WHAT IT DOES
  Pick a model, pick your settings, pick a backend, click Load.
  Zoomies writes a plain PowerShell script matching exactly those
  choices, runs it, and shows the running model in a dashboard with a
  one-click unload. The backend's own app never opens.

  Every filled-in field maps to one part of the generated script.
  Nothing is hidden - click "Preview script" to read it before it runs.


HOW TO START IT
  Double-click  Zoomies.cmd
  or run        python app.py

  Needs Python 3.11+ with tkinter (the standard Windows installer
  includes it). No pip installs, no build step, no other dependencies.


WHERE THINGS LIVE
  Zoomies never writes into this project folder. Everything it keeps
  goes to  %LOCALAPPDATA%\Zoomies :

    config.json    your preferences
    session.json   what is running, and what Zoomies created
    scripts\       every generated script, newest 20 kept
    logs\          matching log for each script, newest 20 kept

  Use "Open folder" and "Open log" in the Output pane to get to them.


THE TWO BACKENDS

  OLLAMA
    Models come from Ollama's own registry, so the folder box is greyed
    out - Ollama manages those files itself. Sampling settings are
    applied through a temporary tag (see below).

  UNSLOTH STUDIO
    Models come from the running Studio (its own API lists them, with
    sizes and quants), plus any .gguf files in the folder you pick.

    Unsloth has the mirror-image of Ollama's problem. Its load API takes
    context length, GPU layers and parallel slots, but has no field for
    temperature. Sampling is pinned by the `unsloth run` command line
    instead - which only happens when Zoomies is the one starting the
    server. So:

      Studio not running -> Zoomies starts it and pins everything.
      Studio already running -> Zoomies loads into it, and tells you
                                plainly which settings could not be
                                applied. Stop the server first if you
                                need them.

    Unloading a model leaves Studio running. Zoomies only ever shuts the
    whole server down if it started it, because "unsloth studio stop"
    stops every server on the machine, not just ours.


THE OLLAMA TEMPORARY TAG - READ THIS ONE
  Ollama has no way to attach sampling settings to a model at launch.
  "ollama run" takes no such flags, "/set parameter" only lives inside
  the interactive chat session, and API options apply to a single
  request. This is a known gap - ollama/ollama issue #5362, open since
  June 2024.

  So when you load an Ollama model WITH settings, Zoomies creates a
  temporary derived tag:

      qwen3.8:27b-q4_K_M   ->   qwen3.8:27b-q4_K_M-zoomies

  and deletes it again when you unload.

  This does NOT duplicate the model. Ollama stores weights as
  content-addressed blobs, and the derived tag points at the very same
  blob - Ollama itself reports "using already created layer sha256:..."
  while creating it. The real cost is about 1 KB of manifest. Measured
  on this machine, a full load-and-unload cycle changes the blob folder
  by exactly 0 bytes.

  If you load with NO settings filled in, no tag is created at all -
  the base model is loaded as-is.

  Five conditions must ALL hold before Zoomies deletes anything:
    1. the name ends in "-zoomies"
    2. Zoomies' own records say it created it
    3. it did not already exist before Zoomies made it
    4. it is not currently loaded
    5. it is not a base model name
  Base model names never enter a delete code path at all. The generated
  script re-checks the "-zoomies" suffix itself, so it is safe to read,
  keep, or run on its own.

  If Zoomies is killed mid-run, the next start offers to clean up any
  leftover tag, and "Temporary tags..." shows you what exists at any
  time.


APPLY OPTIMAL SETTINGS
  Click it and Zoomies looks up that model's recommended sampling
  settings in the Unsloth documentation and fills them in. The line
  underneath says exactly where each number came from - page, section,
  mode and line number - so you can always check it.

  HOW IT FINDS THE RIGHT PAGE
  Unsloth publishes an index of every docs page at
  unsloth.ai/docs/llms.txt, and the page's web address is the model
  family name:

      qwen3.8:27b-q4_K_M     ->  qwen3.8     ->  .../models/qwen3.8.md
      gemma4:12b-it-q4_K_M   ->  gemma4      ->  .../models/gemma-4.md
      ministral-3:14b-...    ->  ministral3  ->  .../tutorials/ministral-3.md

  Take the model name up to the colon, drop the punctuation, and it
  matches. Zoomies then downloads that one page (about 30 KB) instead of
  the whole documentation set.

  WORKED OUT ONCE, THEN KEPT
  The first lookup for a model takes about a fifth of a second. The
  answer is then saved to
      %LOCALAPPDATA%\Zoomies\cache
esolved.json
  and never expires. After that, clicking Apply - or switching modes -
  is instant and needs no internet at all, even if the cache is wiped.
  Press "Re-read docs" when you want Zoomies to go and look again,
  which is the only thing that overwrites a saved answer.

  Blue numbers came from the docs. Numbers you type turn white, and
  Zoomies asks before overwriting anything you changed yourself.

  The Mode dropdown appears when a model documents more than one set of
  numbers (Thinking vs Instruct, Instruct vs Reasoning). Switching it
  re-reads that column. It only changes sampling numbers; it does not
  turn thinking on or off by itself - that is the Reasoning dropdown.

  REASONING
  The Reasoning dropdown offers exactly what the model's docs page says,
  because models do this differently:

      qwen3.8       effort level: xhigh (default), medium, low, none
      gemma4        on / off
      ministral-3   cannot be switched - "separate model": its Reasoning
                    version is a different download

  Reasoning and Mode move together so they never contradict each other:
  choosing Instruct switches reasoning off, and choosing a reasoning level
  switches Mode back to Thinking.

  Unsloth applies the choice when the model loads. On Ollama the dropdown
  is greyed out: Ollama takes reasoning per request, so the app sending
  the prompt (jobbuddy, for example) decides, not the launcher.

  IT WILL LEAVE THE BOXES EMPTY RATHER THAN GUESS. If the docs have no
  page for your model you get an empty form and a list of pages to pick
  from, not a near-match. A wrong temperature looks exactly like a right
  one; an empty box does not. Your pick is remembered for that model.

  The one exception is an older release of the same family: if there is
  no gemma9 page but gemma4 exists, Zoomies uses gemma4's settings and
  says so in orange, because sampling defaults rarely swing much between
  versions of a family. Check them before relying on them.

  Context length is capped when the docs quote something your cards
  cannot hold - they often say 262,144, which no consumer GPU can fit
  beside the weights. The ceiling is worked out from the GPUs actually
  present (Zoomies detects them; this machine has two RX 6800 XTs, about
  32 GB total), not from a fixed number. The note under the settings
  says what it did and why. Raise it if you want.


GREYED-OUT FIELDS
  A greyed field means the selected backend cannot honour that setting,
  and the line under the settings says which ones. Zoomies will never
  quietly accept a value and then ignore it.

  An empty field means "do not pass this setting" - which is not the
  same as passing zero.


NOTES FOR THIS MACHINE
  - The only ollama command Zoomies ever runs is "ollama serve".
    Every other ollama subcommand launches the Ollama tray app;
    everything else here goes over HTTP instead.
  - Your llama.cpp build is Vulkan, not CUDA. Settings copied from
    CUDA-oriented guides (--flash-attn, -ngl) may be ignored or slower.
  - You have TWO RX 6800 XTs, about 32 GB of VRAM in total. Each card is
    16 GB on its own, so a 16.5 GB model like qwen3.8:27b-q4_K_M does not
    fit on a single card - Ollama will spread it or offload part of it.
    Watch the "VRAM in use" figure.
  - There is a stale NVIDIA RTX 4090 entry in the Windows registry from a
    card that is no longer installed. Zoomies ignores it: it counts only
    adapters Windows reports as present.
  - Model docs often list a 262,144 context. Nothing here reaches that;
    8192-65536 is the realistic range.


THE LIVE PANEL
  Once a model is answering, the Live strip shows what it is actually
  doing: prompt-eval speed, time to first token, generation rate (plus a
  3-second average), tokens produced, and how much of the context window
  is used. Your GPUs and their utilisation sit on the right.

  WHERE THE NUMBERS COME FROM
  Both backends run a program called llama-server underneath, whoever
  started them. Zoomies finds every running llama-server and asks it
  directly, twice a second, what it is doing. This works even when the
  model was launched by another app - jobbuddy, for example, starts
  Ollama with its log switched off, so a log-reading monitor would see
  nothing at all.

  Speeds are worked out from how many tokens were added between two
  checks, so time-to-first-token is accurate to about half a second, and
  a request shorter than half a second can be missed entirely.

  If no llama-server can be reached, Zoomies falls back to reading the
  backend's log file instead, when one exists:

    Ollama    %LOCALAPPDATA%\Ollama\server.log      one file
    Unsloth   ~\.unsloth\studio\logs\llama-server\  a new file per run

  The History tab keeps the last 25 requests. A request is filed when it
  finishes, when the next one starts, or once it has been quiet for five
  seconds.

  GPU numbers come from Windows' own "GPU Engine" counters, the same
  source Task Manager reads, and show the highest engine per card rather
  than the sum (summing would double count engines running in parallel).
  Cards are named through DXGI, and two identical cards are told apart by
  the last four digits of their subsystem ID.


STATUS
  v0    Ollama backend, manual settings, load/unload, dashboard.  DONE
  v0.5  the optimizer - recommended settings from the live docs.  DONE
  v1    Unsloth Studio backend, both radio buttons live.          DONE
  v1.5  live tokens/sec, TTFT, context bar and GPU utilisation.   DONE

  Zoomies now covers everything the separate Ollama Monitor did, for
  both backends rather than just Ollama.

KNOWN ROUGH EDGE
  Unsloth reports a cached model's size as the whole downloaded folder,
  which can include more than one quantisation. Qwen3.8-27B shows as
  37.9 GB there but 16.5 GB in Ollama. Zoomies uses that figure to pick
  a context ceiling, so for those models the suggested context comes out
  lower than it needs to be. Raise it by hand if you know better.
