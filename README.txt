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


THE TWO LAYOUTS
  The same Zoomies, drawn two ways. Both do everything; pick whichever
  suits the screen you have.

  CLASSIC   one wide window with every panel stacked: model, settings,
            Loaded, and a Live pane with History, Output and Processes
            as tabs. What Zoomies has always looked like.

  COMPACT   a slim window meant to sit docked beside whatever you are
            working in. One view at a time - Setup, Monitor, History,
            Processes - picked from the icons along the bottom or with
            Ctrl+1 to Ctrl+4. It opens on Setup, or on Monitor when a
            model is already running, and jumps to Monitor when you
            launch. A launch that fails goes back to Setup, where the
            error sits beside the Launch button.

            While a model loads, Monitor shows how long it has taken,
            the log as it arrives, and a Cancel button. Cancelling a
            llama.cpp load ends the script and the server under it, the
            same way Unload stops it. Cancelling an Ollama load ends
            only the script, because it may have started Ollama's own
            server - Ollama can still finish loading in the background,
            in which case the model simply appears as loaded, and any
            temporary tag it made is cleaned up like any other.

            Everything the classic top bar held lives in the "..."
            menu: always on top, unload on exit, one model at a time,
            the model folder, download, temporary tags, Sync opencode,
            the log and scripts folders, Exit, and Unload all and exit.

            It remembers where you put it and opens there next time.
            The first time, it docks to the right edge of the screen.

  SWITCHING
  "Try compact layout" in the classic top bar, or "Switch to classic
  layout" in the compact menu. Zoomies reopens in the other layout,
  which takes a second; loaded models keep running, and whatever you
  had typed in the form comes with it. The choice is remembered.
  From a command line:  python app.py --compact   or   --classic


WHERE THINGS LIVE
  Zoomies never writes into this project folder. Everything it keeps
  goes to  %LOCALAPPDATA%\Zoomies :

    config.json    your preferences, and which layout you last used
    session.json   what is running, and what Zoomies created
    presets.json   measured settings per model - see PRESETS below
    history.json   the last 200 requests - see LIVE NUMBERS below
    handoff.json   only while switching layouts; deleted once read
    scripts\       every generated script, newest 20 kept
    logs\          matching log for each script, newest 20 kept
    cache\         docs pages, model cards, and saved settings answers

  "Open folder" and "Open log" get to them: in the Output pane in the
  classic layout, in the "..." menu in the compact one.


THE TWO BACKENDS

  OLLAMA
    Models come from Ollama's own registry, so the folder box is greyed
    out - Ollama manages those files itself. Sampling settings are
    applied through a temporary tag (see below). Good for trying models
    out, and what jobbuddy uses.

  LLAMA.CPP
    The engine underneath Ollama, run directly. Needs the llama.cpp app
    (its llama.exe) or a llama-server.exe release on this machine.

    Models come from three places: every model Ollama has already
    pulled, the Hugging Face cache (~\.cache\huggingface\hub) where
    llama.cpp downloads to (including .gguf files saved loose in its top
    folder), and any .gguf files in the folder you pick.
    "Rescan" looks again and says how many it found.

    Ollama's models are read straight out of its store. Nothing is
    copied, converted or duplicated - the weights are one file, and
    llama.cpp is handed the path to it - and they keep their Ollama
    names, so the same model reads the same on either backend. That is
    the point: it is how you compare the two on one model.

    "Download..." fetches a new one by its Hugging Face name:

        ggml-org/Qwen3.5-0.8B-GGUF:Q8_0

    Match the quant you tested on Ollama (Q4_K_M, for example).

    Load starts a separate llama.cpp server for that model on the first
    free port from 8080, with every setting on its command line - so
    every field works, and nothing is saved anywhere else. Sampling
    becomes the server's default: an app that sends its own values still
    wins. The Output pane shows the address to point your apps at, and
    the same address opens llama.cpp's own chat page.

    Integrated graphics are left out automatically. They report system
    memory as VRAM, so llama.cpp would happily put layers there, and
    every token would crawl.

    Models loaded in the llama.cpp app also show up under Loaded, marked
    as not started by Zoomies, and Unload asks the app to unload them.
    Zoomies does not load into the app itself: the app takes each model's
    flags from a saved presets file, so per-load settings would mean
    changing that file permanently.

    So does any other llama.cpp server running on this machine, however it
    was started. Every name under Loaded, and in History, is what the
    server itself says it is holding, not what Zoomies remembers starting
    - ports get reused, and a remembered name goes stale the moment one
    is.


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


FILL FROM DOCS
  Picking a model fills in its recommended settings straight away - the
  "Fill from docs" button does the same on demand. The line underneath
  says exactly where each number came from, so you can always check it.

  A new pick clears what the previous model's docs or preset filled in,
  so its numbers never sit under the new model's name; anything you
  typed yourself stays. When nothing is found, the status line says so,
  and pressing "Fill from docs" offers the list of pages to choose from.
  The button is greyed out while a preset is active, because picking the
  preset already filled in everything the docs have (see PRESETS).

  WHERE IT LOOKS, BEST FIRST
    1. The Unsloth documentation page for that model.
    2. The publisher's own Hugging Face model card - found through the
       model's repo, the Hugging Face link stored inside the model file,
       the original a GGUF was made from, or a search that only accepts
       exactly the same model family.
    3. The Unsloth page for an earlier release of the same family.
    4. The settings packaged inside an Ollama model (set by whoever
       published it on Ollama, not necessarily its authors).

  On a model card, numbers only count inside a passage that calls itself
  a recommendation - never in a code example or a benchmark note. Cards
  often list the settings their benchmarks ran with right beside the
  real advice (Ornith's lists temperature 1.0 for its benchmarks), and
  those are not advice.

  HOW IT FINDS THE UNSLOTH PAGE
  Unsloth publishes an index of every docs page at
  unsloth.ai/docs/llms.txt, and the page's web address is the model
  family name:

      qwen3.8:27b-q4_K_M     ->  qwen3.8     ->  .../models/qwen3.8.md
      gemma4:12b-it-q4_K_M   ->  gemma4      ->  .../models/gemma-4.md
      ministral-3:14b-...    ->  ministral3  ->  .../tutorials/ministral-3.md

  Take the model name up to the colon (or the size, for a Hugging Face
  name), drop the punctuation, and it matches.

  WORKED OUT ONCE, THEN KEPT
  The answer for a model family is saved to
      %LOCALAPPDATA%\Zoomies\cache\resolved.json
  and never expires. After that, picking the model, "Fill from docs" or
  switching modes is instant and needs no internet at all - only the
  first pick of a new family goes online. Press "Re-read docs" when you
  want Zoomies to go and look again, which is the only thing that
  overwrites a saved answer.

  Blue numbers came from a source. Numbers you type turn white, and
  Zoomies asks before "Fill from docs" overwrites anything you changed
  yourself.
  Switching Mode or Reasoning updates the other numbers but keeps yours.

  The Mode dropdown appears when a model documents more than one set of
  numbers (Thinking vs Instruct, general vs coding). Switching it
  re-reads that set.

  REASONING
  The Reasoning dropdown offers exactly what the model's chat template
  accepts - the template inside the .gguf, or the file Extra flags name
  with --chat-template-file. It is read as soon as you pick a model,
  because models do this differently:

      qwen3.6, ornith, gemma4, laguna   off / on
      qwen3.8        off / low / medium / xhigh (xhigh is its default)
      granite4.2     off / low-effort / on
      muse-glimmer   low / medium / high / xhigh - no off switch
      ministral-3    cannot be switched - "separate model": its Reasoning
                     version is a different download

  Where a template takes a value without listing the allowed ones (Muse
  Glimmer), the list comes from the model's docs, and reasoning.py says
  which page. When Zoomies cannot tell what a model accepts - no template
  in the file, or an undocumented value - the dropdown says "unknown" and
  the Output pane says why, rather than offering a guess. Switches a
  template has but its docs never mention (Nemotron Lightning's
  medium_effort) are reported there too, and not offered.

  Reasoning and Mode move together so they never contradict each other:
  choosing Instruct switches reasoning off, and choosing a reasoning level
  switches Mode back to Thinking.

  llama.cpp applies the choice when the server starts, as the server's
  default. An app that sends its own setting with a request still wins -
  that is how opencode's dropdown works. On Ollama the dropdown is greyed
  out: Ollama takes reasoning per request, so the app sending the prompt
  (jobbuddy, for example) decides, not the launcher.

  OPENCODE
  "Sync opencode..." (top right) gives every model in opencode's config
  that runs on a Zoomies port the same levels, as opencode "variants" -
  so opencode's reasoning dropdown offers off / low / medium / xhigh for
  Qwen3.8, off / on for Qwen3.6, and so on. It also hides the low /
  medium / high opencode invents for other models, which those templates
  ignore, and sets "reasoning" to whether the model can reason at all.

  It also keeps the sampling numbers in step. opencode sends its
  "options" block with every request, and a request's own value beats
  whatever the server was started with, so a number left there quietly
  overrules Zoomies. Where a model's preset holds sampling numbers, they
  are written into that block and anything Zoomies owns but the preset
  does not is removed, leaving the server's value to stand. Where no
  preset holds any - which is most models until you save some - the
  block is left exactly as it is and said so in the summary, because
  silence in a preset means nobody has saved numbers yet, not that
  llama.cpp's own defaults are wanted. Settings Zoomies has no opinion
  on, such as a timeout, are never touched.

  It shows every change and writes nothing until you press "Write
  changes". Only "reasoning", "variants" and the sampling half of
  "options" change - names, limits and comments stay as written - and
  the old file is kept beside it as opencode.jsonc.bak-<date>-<time>. An agent pointing at a
  variant that was replaced (the old "no-think") is moved to the level
  that sends the same settings ("off"). Models it cannot account for are
  listed and left alone. It does not add or remove models. Restart
  opencode afterwards.

  Levels are listed least reasoning first, because opencode runs titles
  and summaries with a model's first variant - so those run with
  reasoning off.

  KV CACHE
  The KV cache dropdown sits beside Parallel: f16 (llama.cpp's default),
  bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl and f32. It starts on q8_0.
  Smaller types let a longer context fit in the same VRAM, at some cost
  in quality.

  llama.cpp will not start with a quantized V cache (the q and iq types)
  while flash attention is off. If Extra flags contain -fa off, Zoomies
  quantizes only K, leaves V at f16, and says so in the launch notes.
  Live shows it as "K q8_0 / V f16". Use -fa on to quantize both.

  llama.cpp applies it when the server starts. On Ollama it is greyed out
  and shows the current value, because Ollama takes it from the
  OLLAMA_KV_CACHE_TYPE environment variable for every model at once - on
  this machine that is q8_0. The value actually in use shows in the Live
  panel and in History either way.

  VRAM
  Under the settings, Zoomies estimates what a llama.cpp load will need
  on each card before you load it, and updates as you change Context,
  KV cache, Parallel or Extra flags:

      RX 6800 XT (1DA2) 3.7 + 9.5 = 13.2 / 16 GB  ->  fits - 2.8 GB spare

  That is: other apps + this model = total, per card, because layers
  are split across the cards and one can overflow while the total fits.
  Green fits, amber leaves less than 0.5 GB spare, red will spill.
  "Use largest context" fills in the biggest context that still leaves
  0.5 GB free on every card.

  It is worked out from the model file's header - layer sizes, which
  layers keep a KV cache (every 4th on Qwen3.5/3.6/3.8, a few on
  Nemotron, a sliding window on Gemma 4, Muse Glimmer and Laguna), the
  KV cache type, batch size and flash attention - so it is instant and
  never touches the cards. Checked against loads measured on this
  machine, it is within about 0.25 GB per card. Flash attention off is
  the least certain part: its scratch memory grows with context.

  The sliders are what everything else already holds on each card. They
  follow Windows' own counters (the model loaded now is left out when
  "One model at a time" will unload it) until you drag one; "Read usage
  now" puts them all back on the measured value.

  SPILL ALERT
  When a card runs out, Windows does not fail the load - it quietly puts
  the rest in system RAM and the model runs several times slower. Zoomies
  watches every llama.cpp and Ollama server's memory, and when more than
  0.75 GB of it sits in system RAM it says so: a popup, a line in the
  Output pane, and "SPILLING INTO SYSTEM RAM" in red on the Live panel's
  GPU line, which also shows each card's memory in use.

  IT WILL LEAVE THE BOXES EMPTY RATHER THAN GUESS. If no source has
  settings for your model you get an empty form, a list of Unsloth pages
  to pick from, and a "Search the web" button - not a near-match. A wrong
  temperature looks exactly like a right one; an empty box does not. A
  page you pick is remembered for that model.

  Context length is capped when a source quotes something your cards
  cannot hold - they often say 262,144, which no consumer GPU can fit
  beside the weights. The ceiling is worked out from the GPUs actually
  present (this machine has two RX 6800 XTs, about 32 GB total), not
  from a fixed number. The note under the settings says what it did and
  why. Raise it if you want.


PRESETS
  "Fill from docs" fills in what a model's authors recommend.
  A preset fills in what was actually measured on this machine. They
  answer different questions: the docs know the model, only a benchmark
  knows your cards.

  The Preset dropdown lists the presets saved for the selected model on
  the selected backend, and is greyed out when there are none.

  Picking one gives you both: the docs' numbers underneath, the
  preset's on top. A preset usually holds the load settings - context,
  layers, KV cache, flags, reasoning - and no sampling numbers, so the
  docs fill in temperature, top P and the rest, and wherever both have
  a value the preset wins, because it was measured on this machine. The
  note under the settings says which fields came from where.

  The preset stays in charge until you pick another model or press
  Clear: switching Mode or Reasoning re-reads the docs and puts the
  preset's values back on top, so its measured context is never
  replaced by the docs' one. A preset that
  switches reasoning off gets the docs' Instruct numbers. Values arrive
  blue either way, and anything you typed yourself (white) is kept.

  SAVING ONE
  "Save..." beside the dropdown ("Save current" in the compact layout)
  writes the form as a preset for the selected model on the selected
  backend, and asks for a name. Everything filled in is saved - what
  you typed, what the docs filled in and what a preset put there -
  because that is what Launch would use. Empty fields are left out, so
  is anything this backend greys out, and a preset saved on one backend
  is never offered on the other.

  Saving under the name of an existing preset replaces its settings and
  keeps its note, which is how a measured preset gains sampling numbers
  without losing what the benchmark said. Any other name adds a preset,
  noted as saved from the form with the date. The preset in use is
  offered as the default name, so re-saving the one you are working
  from is the quickest path.

  A preset records the backend it was measured on. llama.cpp flags mean
  nothing to Ollama, so an Ollama preset is never offered for a
  llama.cpp model or the other way round. Anything the current backend
  cannot honour is named in the status line rather than dropped quietly.

  Presets live in %LOCALAPPDATA%\Zoomies\presets.json, not in this
  project folder, because the fastest settings depend on the hardware:
  the same model on different cards wants different flags. Copying that
  file to another machine copies numbers that were never true there.

  Models are matched on a normalised name, so the same weights match
  however they arrive - as an Ollama tag, a .gguf path, or a Hugging
  Face name:

      ornith:35b-q4_K_M              -> ornith-35b-q4-k-m
      C:\models\ornith-35b-Q4_K_M.gguf -> ornith-35b-q4-k-m

  The file is plain JSON and safe to edit by hand, and hand-written
  entries survive a save from the app. Each entry has a name, the
  backend it was measured on, a settings block in the same shape the
  form uses, and a note saying where the numbers came from.


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
  - llama.cpp here is the Vulkan build, not CUDA. Settings copied from
    CUDA-oriented guides may be ignored or slower.
  - You have TWO RX 6800 XTs, about 32 GB of VRAM in total. Each card is
    16 GB on its own, so a 16.5 GB model like qwen3.8:27b-q4_K_M does not
    fit on a single card - it is spread across both. Watch the "VRAM in
    use" figure.
  - There is a stale NVIDIA RTX 4090 entry in the Windows registry from a
    card that is no longer installed. Zoomies ignores it: it counts only
    adapters Windows reports as present.


THE LIVE PANEL
  Once a model is answering, the Live strip shows what it is actually
  doing: prompt-eval speed, time to first token, generation rate, tokens
  produced, KV cache type, and how much of the context window is used.
  Your GPUs and their utilisation sit on the right.

  WHERE THE NUMBERS COME FROM
  Both backends run llama.cpp underneath, whoever started them. Zoomies
  finds every running llama.cpp server and asks it directly what it is
  doing - twice a second, five times a second while a request runs. This
  works even when another app launched the model - jobbuddy, for
  example, starts Ollama with its log switched off, so a log-reading
  monitor would see nothing at all.

  Speeds are worked out from how many tokens were added between two
  checks. When a request's prompt and first token both land between two
  checks, time to first token shows as an upper bound, like "<0.20s".

  If no server can be reached, Zoomies falls back to reading Ollama's
  log, %LOCALAPPDATA%\Ollama\server.log, when it exists.

  History keeps the last 200 requests, in history.json, so the numbers
  you measured this morning are still there tomorrow. A request is filed
  when it finishes, when the next one starts, or once it has been quiet
  for five seconds; the file is written at most every few seconds, and
  once more on the way out.

  In the compact layout History groups requests by day, filters by model
  with the chips at the top, and shows one line each - time, model,
  backend, generation speed, context used. Open a row for its time to
  first token, the prompt and output breakdown and the KV type. "Export
  CSV" writes whatever the filter is showing; "Clear" forgets the lot.

  GPU numbers come from Windows' own "GPU Engine" counters, the same
  source Task Manager reads, and show the highest engine per card rather
  than the sum (summing would double count engines running in parallel).
  Cards are named through DXGI, and two identical cards are told apart by
  the last four digits of their subsystem ID.


KEEPING THE MACHINE CLEAN
  Testing one model after another tends to leave memory behind: a
  model still loaded in Ollama or the llama.cpp app, a server Zoomies
  started, or a chat left open in a terminal. Two things deal with it.

  ONE MODEL AT A TIME (next to Load, on by default)
  Before loading, Zoomies unloads everything it can reach - its own
  llama.cpp servers, Ollama's loaded models (including one jobbuddy
  loaded) and the llama.cpp app's models - using the same steps as the
  Unload button, then loads the new one. Untick it to run several
  models side by side.

  PROCESSES
  Lists every llama.exe and ollama.exe on the machine with its memory,
  start time and what started it, and says in words what each one is.
  Anything nothing accounts for is listed first as a LEFTOVER, and the
  tab title - or the Procs icon, in the compact layout - shows how many
  there are:

      a llama.cpp chat or `ollama run` left open in a terminal
      a llama.cpp server started outside Zoomies
      a Zoomies server it lost track of
      an Ollama model whose Ollama server is gone

  "Clean up leftovers" shows the list and ends them once you agree.
  "End selected" ends the rows you pick. Ollama's own server and tray
  app are never ended from here, and before ending anything Zoomies
  checks the process is still the same one it listed - Windows reuses
  process numbers, and a stale row must never hit an unrelated program.


STATUS
  v0    Ollama backend, manual settings, load/unload, dashboard.  DONE
  v0.5  the optimizer - recommended settings from the live docs.  DONE
  v1    live tokens/sec, TTFT, context bar and GPU utilisation.   DONE
  v1.5  settings from model cards and Ollama; llama.cpp backend.  DONE
        (the Unsloth Studio backend was removed in favour of it)
  v2    a compact layout for a window docked at the side of the
        screen, history kept on disk, and Cancel while loading.   DONE

  Zoomies covers everything the separate Ollama Monitor did, for both
  backends rather than just Ollama.
