r"""
Tests for the stable-diffusion.cpp backend.

What matters here is telling a denoiser from a language model, because
getting it wrong is silently expensive in both directions: a diffusion GGUF
offered under llama.cpp is a load that can only fail, and a language model
offered under sd-server would hold VRAM and answer nothing. Neither mistake
raises - the dropdown just lists the wrong thing - so it is checked here.

The GGUF header is faked rather than read from disk. The real files are tens
of gigabytes, and what is being tested is the rule, not the parser.

    python test_sdcpp.py
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gguf
import sdcpp
import state


class FakeHeader:
    """Just the two things is_diffusion_gguf reads."""

    def __init__(self, arch="", tensors=()):
        self.arch = arch
        self.other_bytes = {name: 1 for name in tensors}
        self.layer_bytes = {}


DIFFUSION_TENSORS = (
    "model.diffusion_model.img_in.weight",
    "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight",
    "model.diffusion_model.proj_out.weight",
)
LLM_TENSORS = ("token_embd.weight", "blk.0.attn_q.weight", "output_norm.weight")


class _Tmp(unittest.TestCase):
    """A throwaway directory that goes away again.

    tempfile.mkdtemp() on its own leaks one directory per test, which over a
    few hundred runs is a visible pile under %TEMP%.
    """

    def tmpdir(self):
        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path, True)
        return path


class _PatchGguf(_Tmp):
    """Swaps gguf.read for a lookup over fake headers."""

    def setUp(self):
        self.headers = {}
        self._real = gguf.read
        gguf.read = self._read

    def tearDown(self):
        gguf.read = self._real

    def _read(self, path):
        key = os.path.basename(path).lower()
        if key not in self.headers:
            raise ValueError("no header for %s" % key)
        return self.headers[key]


class Identifying(_PatchGguf):

    def test_a_denoiser_is_recognised_by_its_tensor_names(self):
        # No general.architecture at all, which is what the conversions carry.
        self.headers["qwen-image-2.1-f16.gguf"] = FakeHeader("", DIFFUSION_TENSORS)
        self.assertTrue(sdcpp.is_diffusion_gguf("/m/qwen-image-2.1-F16.gguf"))

    def test_a_language_model_is_not_a_denoiser(self):
        self.headers["qwen3.8-27b.gguf"] = FakeHeader("qwen3", LLM_TENSORS)
        self.assertFalse(sdcpp.is_diffusion_gguf("/m/Qwen3.8-27B.gguf"))

    def test_a_named_architecture_settles_it_without_reading_tensors(self):
        """An arch means llama.cpp has an implementation to dispatch to.

        Guards the case of a future diffusion arch landing in llama.cpp: the
        file would then name itself, and this backend should stop claiming it.
        """
        self.headers["odd.gguf"] = FakeHeader("some-arch", DIFFUSION_TENSORS)
        self.assertFalse(sdcpp.is_diffusion_gguf("/m/odd.gguf"))

    def test_companion_files_are_never_offered_as_the_model(self):
        """The VAE and text encoder live beside the denoiser.

        Rejected on the name before the header is read, so no header is
        registered for them here - reading one would raise.
        """
        for name in ("qwen_image_2.1_vae_bf16.safetensors",
                     "mmproj-Qwen3-VL-8B-F16.gguf",
                     "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
                     "Qwen3.8-27B-Uncensored-draft-Q8_0.gguf"):
            self.assertFalse(sdcpp.is_diffusion_gguf("/m/" + name), name)

    def test_an_unreadable_file_is_not_claimed(self):
        self.assertFalse(sdcpp.is_diffusion_gguf("/m/nothing-here.gguf"))


class FindingTheProgram(_Tmp):
    """sd-server has to be found without depending on one env var.

    A GUI started from a shortcut does not always inherit LOCALAPPDATA. When
    it did not, the lookup returned "" and the app reported sd-server missing
    while it was installed, and every Load wrote a refusal to the log.
    """

    def setUp(self):
        self.env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)

    def _install(self):
        """A fake install under a fake home."""
        home = self.tmpdir()
        folder = os.path.join(home, "AppData", "Local", "stable-diffusion.cpp")
        os.makedirs(folder)
        exe = os.path.join(folder, "sd-server.exe")
        with open(exe, "w", encoding="utf-8") as fh:
            fh.write("x")
        return home, exe

    def test_found_through_localappdata(self):
        home, exe = self._install()
        os.environ["PATH"] = ""
        os.environ["LOCALAPPDATA"] = os.path.join(home, "AppData", "Local")
        self.assertEqual(sdcpp.find_sd_exe(), exe)

    def test_found_with_no_localappdata_at_all(self):
        home, exe = self._install()
        os.environ["PATH"] = ""
        os.environ.pop("LOCALAPPDATA", None)
        os.environ["USERPROFILE"] = home          # what expanduser reads
        self.assertEqual(sdcpp.find_sd_exe(), exe)

    def test_nothing_found_when_it_is_not_installed(self):
        os.environ["PATH"] = ""
        os.environ["LOCALAPPDATA"] = self.tmpdir()
        os.environ["USERPROFILE"] = self.tmpdir()
        self.assertEqual(sdcpp.find_sd_exe(), "")


class ItsOwnDeviceListing(unittest.TestCase):
    """sd-server names its devices differently from llama.cpp.

    llama.cpp writes "Vulkan0: AMD Radeon RX 6800 XT (16368 MiB, ...)";
    sd-server writes "Vulkan0<TAB>AMD Radeon RX 6800 XT". Its help says the
    names it prints are the ones --backend accepts, so these are the ones to
    report - reading llama.cpp's instead assumed two ggml builds enumerate
    the same adapters in the same order.
    """

    LISTING = (
        "ggml_vulkan: Found 3 Vulkan devices:\n"
        "ggml_vulkan: 0 = AMD Radeon RX 6800 XT (AMD proprietary driver)\n"
        "load_backend: loaded Vulkan backend from ggml-vulkan.dll\n"
        "Vulkan0\tAMD Radeon RX 6800 XT\n"
        "Vulkan1\tIntel(R) UHD Graphics 770\n"
        "Vulkan2\tAMD Radeon RX 6800 XT\n"
        "CPU\tIntel(R) Core(TM) i9-14900K\n")

    def setUp(self):
        self.real = sdcpp.subprocess.run
        sdcpp._DEVICES.clear()

        class Result:
            stdout = ItsOwnDeviceListing.LISTING
            stderr = ""

        sdcpp.subprocess.run = lambda *a, **k: Result()

    def tearDown(self):
        sdcpp.subprocess.run = self.real
        sdcpp._DEVICES.clear()

    def test_the_loaders_own_chatter_is_not_read_as_a_device(self):
        """Only the tab separated lines are the listing."""
        self.assertEqual(sdcpp.list_devices("sd-server.exe"), [
            ("Vulkan0", "AMD Radeon RX 6800 XT"),
            ("Vulkan1", "Intel(R) UHD Graphics 770"),
            ("Vulkan2", "AMD Radeon RX 6800 XT"),
            ("CPU", "Intel(R) Core(TM) i9-14900K"),
        ])

    def test_the_integrated_gpu_is_the_one_to_leave_out(self):
        """It sits between the two cards, so the obvious pair is wrong.

        Vulkan0&Vulkan1 would put half the weights on a chip sharing system
        memory. The pair worth using is Vulkan0&Vulkan2.
        """
        be = sdcpp.SdCppBackend()
        be.exe = "sd-server.exe"
        notes = " ".join(be._device_notes())
        self.assertIn("Leave Vulkan1 out", notes)
        self.assertIn("diffusion=Vulkan0&Vulkan2", notes)
        self.assertNotIn("Vulkan0&Vulkan1", notes)

    def test_the_cpu_is_a_backend_but_not_a_card(self):
        """It must not be offered as somewhere to spread the weights."""
        be = sdcpp.SdCppBackend()
        be.exe = "sd-server.exe"
        notes = " ".join(be._device_notes())
        self.assertNotIn("CPU", notes.replace("CPU)", ""))

    def _notes_for(self, model_bytes, card_bytes=16 * (1 << 30)):
        be = sdcpp.SdCppBackend()
        be.exe = "sd-server.exe"

        class Record:
            gguf_path = "C:\\models\\denoiser.gguf"
            id = gguf_path

        real_size, real_card = os.path.getsize, sdcpp._smallest_card
        os.path.getsize = lambda p: model_bytes
        sdcpp._smallest_card = lambda: card_bytes
        try:
            return " ".join(be._device_notes(Record()))
        finally:
            os.path.getsize = real_size
            sdcpp._smallest_card = real_card

    def test_a_model_that_fits_one_card_is_not_sent_to_two(self):
        """Splitting costs speed, so it is only worth suggesting when needed.

        Dividing a 4 GB denoiser over two 16 GB cards would have suggested
        --max-vram 2, which is both pointless and worse than leaving it off.
        """
        notes = self._notes_for(4 * (1 << 30))
        self.assertIn("fits on one card", notes)
        self.assertNotIn("--max-vram", notes)

    def test_a_model_too_big_for_one_card_gets_half_of_itself(self):
        """Half the denoiser per card is what forces an even split."""
        notes = self._notes_for(int(13.5 * (1 << 30)))
        self.assertIn("--max-vram 7", notes)   # 13.5 GB over two cards
        self.assertNotIn("fits on one card", notes)

    def test_the_graph_counts_against_the_card_too(self):
        """A denoiser that only just fits does not fit.

        The file size is the weights alone; the denoising graph wants GBs
        more on the same card, and sd-server fails at the first request
        rather than at load if that was not left spare.
        """
        card = 16 * (1 << 30)
        self.assertIn("--max-vram", self._notes_for(card - (1 << 30), card))

    def test_the_split_advice_says_what_max_vram_is_for(self):
        """--max-vram is what decides the partition, not a safety limit.

        The splitter fills the first card up to the budget and reserves
        nothing for the graph, so leaving it out or setting it high puts
        nearly everything on one card and the request then fails. Saying
        "--split-mode layer" without saying this sends people to a setting
        that looks unsupported.
        """
        be = sdcpp.SdCppBackend()
        be.exe = "sd-server.exe"
        notes = " ".join(be._device_notes())
        self.assertIn("--max-vram", notes)
        self.assertIn("divided by the number of cards", notes)


class Companions(_Tmp):

    def setUp(self):
        self.dir = self.tmpdir()

    def _touch(self, name):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("x")
        return path

    def test_both_companions_are_found_beside_the_model(self):
        model = self._touch("qwen-image-2.1-F16.gguf")
        vae = self._touch("qwen_image_2.1_vae_bf16.safetensors")
        llm = self._touch("Qwen3-VL-8B-Instruct-Q4_K_M.gguf")
        kit = sdcpp.companions_for(model)
        self.assertEqual(kit.vae, vae)
        self.assertEqual(kit.llm, llm)
        self.assertTrue(kit.complete)
        self.assertEqual(kit.missing(), [])

    def test_a_denoiser_on_its_own_reports_both_as_missing(self):
        model = self._touch("qwen-image-2.1-F16.gguf")
        kit = sdcpp.companions_for(model)
        self.assertFalse(kit.complete)
        self.assertEqual(len(kit.missing()), 2)
        self.assertIn("2.1", kit.missing()[0])

    def test_a_vae_from_the_wrong_generation_is_not_accepted(self):
        """Qwen-Image and Qwen-Image 2.1 VAEs are not interchangeable.

        The wrong one decodes to noise rather than failing, so it must not be
        picked up just because the filename says vae.
        """
        model = self._touch("qwen-image-2.1-F16.gguf")
        self._touch("qwen_image_vae.safetensors")        # the 1.x VAE
        kit = sdcpp.companions_for(model)
        self.assertEqual(kit.vae, "")
        self.assertFalse(kit.complete)

    def test_the_mmproj_is_not_mistaken_for_the_text_encoder(self):
        model = self._touch("qwen-image-2.1-F16.gguf")
        self._touch("mmproj-Qwen3-VL-8B-F16.gguf")
        kit = sdcpp.companions_for(model)
        self.assertEqual(kit.llm, "")

    def test_the_projector_is_found_for_editing_by_instruction(self):
        model = self._touch("qwen-image-2.1-F16.gguf")
        self._touch("qwen_image_2.1_vae_bf16.safetensors")
        self._touch("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")
        proj = self._touch("mmproj-Qwen3-VL-8B-Instruct-F16.gguf")
        kit = sdcpp.companions_for(model)
        self.assertEqual(kit.vision, proj)

    def test_a_projector_from_another_model_is_never_used(self):
        """The hub folder also holds gemma's projector for llama.cpp.

        A projector is trained against one encoder. Handing gemma's to a Qwen
        encoder loads without complaint and then conditions on nonsense, so
        the family in the name has to match the encoder that was chosen.
        """
        model = self._touch("qwen-image-2.1-F16.gguf")
        self._touch("qwen_image_2.1_vae_bf16.safetensors")
        self._touch("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")
        self._touch("mmproj-gemma-4-31B-it-F16.gguf")
        kit = sdcpp.companions_for(model)
        self.assertEqual(kit.vision, "")

    def test_the_model_is_still_usable_without_a_projector(self):
        """No projector is not an error - only instruction editing is lost."""
        model = self._touch("qwen-image-2.1-F16.gguf")
        self._touch("qwen_image_2.1_vae_bf16.safetensors")
        self._touch("Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")
        kit = sdcpp.companions_for(model)
        self.assertTrue(kit.complete)
        self.assertEqual(kit.vision, "")

    def test_each_generation_asks_for_its_own_text_encoder(self):
        self.assertEqual(sdcpp.Companions(version="2.1").encoder_name(),
                         "Qwen3-VL-8B-Instruct")
        self.assertEqual(sdcpp.Companions(version="").encoder_name(),
                         "Qwen2.5-VL-7B-Instruct")


class TheBackend(_PatchGguf):

    def setUp(self):
        _PatchGguf.setUp(self)
        self.be = sdcpp.SdCppBackend()
        self.dir = self.tmpdir()

    def _model(self, name="qwen-image-2.1-F16.gguf"):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("x")
        self.headers[name.lower()] = FakeHeader("", DIFFUSION_TENSORS)
        return path

    def _complete(self):
        """A denoiser with its VAE and text encoder, so a launch really runs.

        Without them build_launch refuses and writes an exit 2 script, and a
        test that looks for a flag in the script passes because the script
        has no flags in it at all.
        """
        path = self._model()
        for name in ("qwen_image_2.1_vae_bf16.safetensors",
                     "Qwen3-VL-8B-Instruct.gguf"):
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
                fh.write("x")
        return path

    def test_only_denoisers_are_listed(self):
        self._model()
        llm = os.path.join(self.dir, "Qwen3.8-27B.gguf")
        with open(llm, "w", encoding="utf-8") as fh:
            fh.write("x")
        self.headers["qwen3.8-27b.gguf"] = FakeHeader("qwen3", LLM_TENSORS)
        labels = [m.label for m in self.be.list_models(self.dir)]
        self.assertEqual(labels, ["qwen-image-2.1-F16"])

    def test_a_listed_model_says_it_makes_images(self):
        self._model()
        self.assertEqual(self.be.list_models(self.dir)[0].capabilities, ("image",))

    def test_language_model_settings_are_all_greyed_out(self):
        """Every one of these is meaningless to a denoiser.

        supports() returning "" is what makes the GUI grey the field and say
        why, rather than take a number and quietly drop it.
        """
        for key in ("context_length", "gpu_layers", "kv_cache", "reasoning",
                    "parallel", "temperature", "top_p", "max_tokens",
                    "keep_alive"):
            self.assertEqual(self.be.supports(key), "", key)
        self.assertTrue(self.be.supports("extra_flags"))

    def test_there_is_no_kv_cache_to_impose(self):
        self.assertIsNone(self.be.fixed_kv_cache())

    def test_an_incomplete_model_refuses_to_start(self):
        """No VAE and no encoder means no launch.

        Starting anyway would take 14 GB of VRAM and fail every request, so
        the script must exit instead of calling the binary.
        """
        path = self._model()
        record = self.be.list_models(self.dir)[0]
        plan = self.be.build_launch(record, {}, {})
        self.assertNotIn("& $exe @a", plan.script_text)
        self.assertIn("exit 2", plan.script_text)
        self.assertTrue(any("denoiser only" in n for n in plan.notes))
        self.assertTrue(path)

    def test_zoomies_keeps_control_of_the_model_paths_and_port(self):
        """Extra flags may not redirect the launch.

        A --port or --vae through the free-text field would leave the
        dropdown and the running server disagreeing about what is loaded.
        """
        self._complete()
        record = self.be.list_models(self.dir)[0]
        plan = self.be.build_launch(
            record,
            {"extra_flags": "--diffusion-fa --listen-port 9999 --vae other.st"},
            {})
        self.assertTrue(any("Ignoring" in n for n in plan.notes))
        self.assertNotIn("9999", plan.script_text)

    def test_a_quoted_flag_is_still_caught_by_the_denied_list(self):
        """Quoting must not be a way round the flags Zoomies owns.

        Extra flags are split with posix=False, which leaves the quotes on
        the token so that a value like "diffusion=a&b" survives intact. That
        made '"--listen-port"' a different string from '--listen-port', and
        the guard let it through.
        """
        self._complete()
        record = self.be.list_models(self.dir)[0]
        plan = self.be.build_launch(
            record, {"extra_flags": '"--listen-port" 9999'}, {})
        self.assertTrue(any("Ignoring" in n for n in plan.notes))
        self.assertNotIn("9999", plan.script_text)

    def test_a_quoted_value_reaches_sd_server_as_written(self):
        """The other half: quoting a value is how --backend is written."""
        self._complete()
        record = self.be.list_models(self.dir)[0]
        plan = self.be.build_launch(
            record,
            {"extra_flags": '--backend "diffusion=Vulkan0&Vulkan2"'}, {})
        self.assertFalse(any("Ignoring" in n for n in plan.notes))
        self.assertIn("diffusion=Vulkan0&Vulkan2", plan.script_text)

    def test_zoomies_own_page_is_handed_to_sd_server(self):
        """The dark theme is only reachable through --serve-html-path.

        sd-server has its own page built into the binary and serves that at /
        unless it is pointed at a file. Building a nicer page and not passing
        the flag means nobody ever sees it.
        """
        self._complete()
        record = self.be.list_models(self.dir)[0]
        page = os.path.join(self.dir, "index.html")
        with open(page, "w", encoding="utf-8") as fh:
            fh.write("<html></html>")
        real, sdcpp.find_web_ui = sdcpp.find_web_ui, lambda: page
        try:
            plan = self.be.build_launch(record, {}, {})
        finally:
            sdcpp.find_web_ui = real
        self.assertIn("--serve-html-path", plan.script_text)
        self.assertIn(page, plan.script_text)

    def test_a_page_of_your_own_is_not_overridden(self):
        """Extra flags win: this one is not a setting Zoomies owns."""
        self._complete()
        record = self.be.list_models(self.dir)[0]
        real, sdcpp.find_web_ui = sdcpp.find_web_ui, lambda: "C:\\ours.html"
        try:
            plan = self.be.build_launch(
                record, {"extra_flags": "--serve-html-path C:\\mine.html"}, {})
        finally:
            sdcpp.find_web_ui = real
        self.assertIn("mine.html", plan.script_text)
        self.assertNotIn("ours.html", plan.script_text)
        self.assertEqual(plan.script_text.count("--serve-html-path"), 1)

    def test_nothing_is_passed_when_the_page_has_not_been_built(self):
        """sdcpp-webui/dist is a build output, so it may simply not be there.

        Passing a path that does not exist would be worse than passing
        nothing: sd-server would serve an empty page instead of its own.
        """
        self._complete()
        record = self.be.list_models(self.dir)[0]
        real, sdcpp.find_web_ui = sdcpp.find_web_ui, lambda: ""
        try:
            plan = self.be.build_launch(record, {}, {})
        finally:
            sdcpp.find_web_ui = real
        self.assertNotIn("--serve-html-path", plan.script_text)

    def test_the_page_has_to_be_a_file_that_is_there(self):
        """find_web_ui checks, rather than trusting the layout."""
        self.assertEqual(
            os.path.basename(sdcpp.find_web_ui() or "index.html"), "index.html")

    def test_the_address_flags_are_the_ones_sd_server_actually_takes(self):
        """sd-server uses --listen-ip/--listen-port, not --host/--port.

        Checked against `sd-server --help` on the master-945-a1ded76 Vulkan
        build. Passing llama.cpp's names put the server on its default port
        while Zoomies watched another one, so it never looked ready.
        """
        self._model()
        vae = os.path.join(self.dir, "qwen_image_2.1_vae_bf16.safetensors")
        llm = os.path.join(self.dir, "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")
        for path in (vae, llm):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("x")
        record = self.be.list_models(self.dir)[0]
        self.be.exe = r"C:\fake\sd-server.exe"      # so the real call is built
        text = self.be.build_launch(record, {}, {}).script_text
        self.assertIn("'--listen-ip'", text)
        self.assertIn("'--listen-port'", text)
        self.assertNotIn("'--host'", text)
        self.assertNotIn("'--port'", text)
        # and the three model files are all handed over
        for flag in ("'--diffusion-model'", "'--vae'", "'--llm'"):
            self.assertIn(flag, text)

    def test_a_loaded_image_server_carries_no_context(self):
        """Context zero is what keeps it out of the chat harness syncs.

        opencode, dsh and hermes all skip a loaded model with no context, so
        an image server can never be written into their model lists.
        """
        self.assertEqual(
            [m for m in self.be.list_loaded() if m.context], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
