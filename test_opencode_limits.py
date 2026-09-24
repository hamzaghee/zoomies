r"""
Zoomies - what Zoomies is allowed to tell opencode about the context.

    python -m unittest test_opencode_limits -v

Nothing is loaded and nothing is contacted: the config is a temporary file,
the "servers" are plain objects, and the two helpers that would go reading
presets and chat templates off this machine are stubbed, so neither
%USERPROFILE%\.config\opencode nor %LOCALAPPDATA%\Zoomies is touched.

The rule held here is the one a session broke on 2026-09-23. opencode never
asks a server how much room it has; it reads `limit.context` out of its own
config and compacts at context minus output. So a number may only come from
a server that is up, and only into the entry for the model that server is
running - anything else is a guess about a window somebody may be sitting
in. The full sync used to fall back to the model's preset when it was down,
which wrote 48k over the 90k a load had just measured and compacted a 90k
session at 37k; the poll that should have put it back compared only the
running servers, so it never looked again while the same server stayed up.
"""

import os
import tempfile
import unittest

import opencode
import reasoning

# A config with two models on one llama.cpp port and one on another that is
# nothing to do with us. The comments and the hand-written "name" are here
# so a write has something to preserve.
CONFIG = """\
// Local models served by Zoomies.
{
  "model": "zoomies/served:35b",
  "provider": {
    "zoomies": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "served:35b": {
          "name": "the one that is up (64k)",
          "limit": { "context": 65536, "output": 16384 }
        },
        "idle:27b": {
          "name": "the one that is not (48k)",
          "limit": { "context": 49152, "output": 12288 }
        }
      }
    },
    "somewhere-else": {
      "options": { "baseURL": "https://api.example.com/v1" },
      "models": {
        "served:35b": { "limit": { "context": 8192, "output": 2048 } }
      }
    }
  }
}
"""


class Server:
    """A llama.cpp server as backends.list_loaded() describes one."""

    def __init__(self, label, context, port=8080, id=None, source="folder"):
        self.label = label
        self.id = id or label
        self.gguf_path = self.id
        self.context = context
        self.endpoint = "http://127.0.0.1:%d" % port
        self.backend = "llamacpp"
        self.source = source


class WithAConfig(unittest.TestCase):
    """A temporary config, and none of the machine behind it."""

    text = CONFIG

    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".jsonc")
        os.close(handle)
        self.write(self.text)
        self.addCleanup(os.unlink, self.path)

    def write(self, text):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def limits(self, text, provider="zoomies"):
        """{model: limit} as the config reads now."""
        root = opencode.parse(text)
        for pid, pnode, _ in root.get("provider").members:
            if pid != provider:
                continue
            models = pnode.get("models")
            return {mid: (node.value or {}).get("limit")
                    for mid, node, _ in (models.members if models else [])}
        return {}


class TheLoadTimeWriter(WithAConfig):
    """plan_limits: what a load is allowed to say."""

    def test_the_running_model_is_told_what_it_is_serving(self):
        plan = opencode.plan_limits(self.path, [Server("served:35b", 90112)])
        self.assertEqual(self.limits(plan.new_text)["served:35b"],
                         {"context": 90112, "output": 22528})

    def test_every_other_entry_is_left_alone(self):
        plan = opencode.plan_limits(self.path, [Server("served:35b", 90112)])
        before, after = self.limits(plan.text), self.limits(plan.new_text)
        self.assertEqual(after["idle:27b"], before["idle:27b"])
        self.assertEqual(len(plan.changed), 1)

    def test_a_port_with_nothing_on_it_is_left_alone(self):
        plan = opencode.plan_limits(self.path, [])
        self.assertEqual(plan.new_text, plan.text)
        self.assertEqual(plan.changed, [])

    def test_a_server_on_another_port_says_nothing_about_this_one(self):
        plan = opencode.plan_limits(
            self.path, [Server("served:35b", 90112, port=8081)])
        self.assertEqual(plan.new_text, plan.text)

    def test_a_provider_that_is_not_a_local_server_is_never_touched(self):
        plan = opencode.plan_limits(self.path, [Server("served:35b", 90112)])
        self.assertEqual(self.limits(plan.new_text, "somewhere-else"),
                         self.limits(plan.text, "somewhere-else"))

    def test_an_entry_keyed_by_the_file_is_found_by_the_label(self):
        """A config may name a loose .gguf by its path or by its name."""
        self.write(CONFIG.replace('"served:35b": {\n          "name"',
                                  '"C:/models/served.gguf": {\n          "name"'))
        server = Server("served:35b", 90112, id="C:/models/served.gguf")
        plan = opencode.plan_limits(self.path, [server])
        self.assertEqual(self.limits(plan.new_text)["C:/models/served.gguf"],
                         {"context": 90112, "output": 22528})

    def test_an_entry_with_no_limit_block_gets_one(self):
        self.write(CONFIG.replace(
            ',\n          "limit": { "context": 65536, "output": 16384 }', ""))
        plan = opencode.plan_limits(self.path, [Server("served:35b", 90112)])
        self.assertEqual(self.limits(plan.new_text)["served:35b"],
                         {"context": 90112, "output": 22528})

    def test_a_context_already_right_is_not_written_again(self):
        plan = opencode.plan_limits(self.path, [Server("served:35b", 65536)])
        self.assertEqual(plan.changed, [])

    def test_what_is_written_is_said_out_loud(self):
        plan = opencode.plan_limits(self.path, [Server("served:35b", 90112)])
        self.assertIn("65536 -> 90112", plan.changed[0])


class TheFullSync(WithAConfig):
    """plan_sync: the same rule, from the half that also writes reasoning
    levels and sampling numbers."""

    def setUp(self):
        super().setUp()
        # The parts that would read this machine's presets and templates.
        # What they answer does not change the decision under test.
        self.patch(opencode, "_spec_for_config", lambda model: reasoning.Spec())
        self.patch(opencode, "_sampling_for_config", lambda model: ({}, ""))
        self.patch(opencode, "_preset_max_tokens", lambda candidates: "")

    def patch(self, module, name, value):
        original = getattr(module, name)
        setattr(module, name, value)
        self.addCleanup(setattr, module, name, original)

    def models(self):
        return [Server("served:35b", 0), Server("idle:27b", 0)]

    def test_a_preset_never_speaks_for_a_server_that_is_down(self):
        """The bug: a full sync with nothing running used to write each
        model's preset context, talking a live window down to a number
        nobody had launched."""
        plan = opencode.plan_sync(self.path, models=self.models(), loaded=[])
        self.assertEqual(self.limits(plan.new_text), self.limits(plan.text))
        self.assertEqual([c for c in plan.changed if "limit:" in c], [])

    def test_a_running_server_still_writes_its_own_entry(self):
        plan = opencode.plan_sync(self.path, models=self.models(),
                                  loaded=[Server("served:35b", 90112)])
        after = self.limits(plan.new_text)
        self.assertEqual(after["served:35b"], {"context": 90112, "output": 22528})
        self.assertEqual(after["idle:27b"], self.limits(plan.text)["idle:27b"])


class TheStamp(WithAConfig):
    """config_stamp: what the poll compares so that a config changed by
    somebody else is read again, rather than standing until a model is
    loaded or unloaded."""

    def test_a_missing_config_has_no_stamp(self):
        self.assertIsNone(opencode.config_stamp(self.path + ".nope"))

    def test_a_config_nobody_touched_keeps_its_stamp(self):
        self.assertEqual(opencode.config_stamp(self.path),
                         opencode.config_stamp(self.path))

    def test_the_stamp_moves_when_the_file_does(self):
        before = opencode.config_stamp(self.path)
        self.write(CONFIG.replace('"name": "the one that is up (64k)",\n', ""))
        self.assertNotEqual(opencode.config_stamp(self.path), before)

    def test_a_context_swapped_for_one_the_same_length_is_still_seen(self):
        """65536 and 90112 are the same eight bytes written a moment apart,
        which is what the full sync does to a live number."""
        before = opencode.config_stamp(self.path)
        self.write(CONFIG.replace("65536", "90112"))
        self.assertNotEqual(opencode.config_stamp(self.path), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
