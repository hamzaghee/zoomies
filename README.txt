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
      %LOCALAPPDATA%\Zoomies\cacheesolved.json
  and never expires. After that, clicking Apply - or switching modes -
  is instant and needs no internet at all, even if the cache is wiped.
  Press "Re-read docs" when you want Zoomies to go and look again,
  which is the only thing that overwrites a saved answer.

  Blue numbers came from the docs. Numbers you type turn white, and
  Zoomies asks before overwriting anything you changed yourself.

  The Mode dropdown appears when a model documents more than one set of
  numbers (Thinking vs Instruct, Instruct vs Reasoning). Switching it
  re-reads that column.

  IT WILL LEAVE THE BOXES EMPTY RATHER THAN GUESS. If the docs have no
  page for your model you get an empty form and a list of pages to pick
  from, not a near-match. A wrong temperature looks exactly like a right
  one; an empty box does not. Your pick is remembered for that model.

  The one exception is an older release of the same family: if there is
  no gemma9 page but gemma4 exists, Zoomies uses gemma4's settings and
  says so in orange, because sampling defaults rarely swing much between
  versions of a family. Check them before relying on them.

  Context length is always capped to 32,768 even when the docs say
  262,144, because the full window will not fit in 16 GB. The note under
  the settings says so when it happens. Raise it if you have the room.


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
  - qwen3.8:27b-q4_K_M is 16.5 GB and your RX 6800 XT has 16 GB. It
    will not fit at useful context. Watch the "VRAM in use" figure.
  - Model docs often list a 262,144 context. That is not achievable on
    16 GB; set something realistic like 8192-32768.


STATUS
  v0    Ollama backend, manual settings, load/unload, dashboard.  DONE
  v0.5  the optimizer - recommended settings from the live docs.  DONE
  next  the Unsloth Studio backend (the Unsloth radio button)
  then  live tokens/sec, time-to-first-token and GPU metrics
