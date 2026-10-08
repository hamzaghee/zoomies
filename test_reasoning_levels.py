r"""
Zoomies - the reasoning levels read out of a chat template.

    python -m unittest test_reasoning_levels -v

No model is loaded and no .gguf is opened: every test drives
reasoning.detect() with template text directly, so the whole suite runs
away from this machine's model folder. Each fixture is an excerpt of the
template that model ships with, cut down to the lines about thinking; only
the long prose strings inside them are shortened, because the jinja around
them is the whole thing under test.

The case this was written for is gpt-oss, found blind on 2026-10-01. Its
harmony template reads reasoning_effort, so detect() knew it thinks, but
two things stopped it offering a level and the Reasoning dropdown showed
the reason instead of low / medium / high:

  * the template never lists its values, and DOCUMENTED had no gpt-oss
    entry to read them from; and
  * it writes its default as a statement - `{%- if x is not defined %}{%-
    set x = "medium" %}` - which _default_value did not recognise, so even
    with the list it would have reported no default.

Both halves are needed, and the tests below hold each one on its own so
removing either is a failure rather than a quieter dropdown. The other
families are here as regression fixtures: the fix must not move them.
"""

import unittest

import reasoning

# gpt-oss-20b, from the harmony template inside the .gguf: the comment that
# documents the variable, the default written as a statement, and the line
# that renders it. There is no enable_thinking anywhere in it.
HARMONY = '''\
{#-
  - "reasoning_effort": A string that describes the reasoning effort, defaults to "medium".
-#}
{%- if model_identity is not defined %}
    {%- set model_identity = "You are ChatGPT, a large language model trained by OpenAI." %}
{%- endif %}
{{- model_identity + "\\n" }}
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\\n\\n" }}
<|start|>assistant
'''

# Qwen3.8-27B: a switch, a list the template enforces itself, and a default
# written the expression way (`|default`).
QWEN38 = """\
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'high' %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '.') }}
    {%- endif %}
{%- endif %}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- endif %}
"""

# Ornith-1.5-9B: the switch on its own, which is all Qwen3.6, Gemma 4 and
# GLM have too.
ORNITH = """\
{{- '<|im_start|>assistant\\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- endif %}
"""

# Granite 4.2: a boolean effort flag beside the switch, and reasoning_effort
# only as a way of setting that flag.
GRANITE = """\
{%- set low_effort = low_effort if low_effort is defined else False %}
{%- if reasoning_effort is defined %}
    {%- set low_effort = reasoning_effort == 'low' %}
{%- endif %}
{%- if enable_thinking is defined and enable_thinking %}
    {{- '<think>' }}
{%- endif %}
"""

# Muse Glimmer: a string knob with no list and no off switch, the case
# DOCUMENTED was added for in the first place.
MUSE = """\
{%- set strength = reasoning_strength|default('medium') %}
{{- 'Reasoning strength: ' + strength }}
<think>
"""

# A model id for each, flattened the way _documented flattens it. The
# llama.cpp backend uses the full path as the id, so the real gpt-oss id is
# a path - the match has to survive that.
GPTOSS_ID = (r"C:\Users\Hamza\.cache\huggingface\hub"
             r"\gpt-oss-20b-UD-Q8_K_XL.gguf")


class GptOss(unittest.TestCase):
    """The dropdown gpt-oss could not fill."""

    def spec(self, text=HARMONY, model_id=GPTOSS_ID):
        return reasoning.detect(text, model_id)

    def test_the_three_documented_levels_are_offered(self):
        self.assertEqual(self.spec().levels, ("low", "medium", "high"))

    def test_the_template_default_is_medium(self):
        self.assertEqual(self.spec().default, "medium")

    def test_nothing_is_reported_as_a_problem(self):
        spec = self.spec()
        self.assertEqual(spec.problem, "")
        self.assertTrue(spec.usable)
        self.assertTrue(spec.thinks)

    def test_each_level_is_the_kwargs_that_produce_it(self):
        self.assertEqual(self.spec().kwargs, {
            "low": {"reasoning_effort": "low"},
            "medium": {"reasoning_effort": "medium"},
            "high": {"reasoning_effort": "high"},
        })

    def test_there_is_no_off_level(self):
        # Nothing in the harmony template switches thinking off, so the
        # dropdown must not pretend one of these levels does.
        spec = self.spec()
        self.assertEqual(spec.off, "")
        self.assertNotIn("off", spec.levels)

    def test_where_the_values_came_from_is_said_out_loud(self):
        notes = " ".join(self.spec().notes)
        self.assertIn("reasoning_effort values from", notes)
        self.assertIn("gpt-oss model card", notes)

    def test_the_id_being_a_full_path_does_not_hide_the_match(self):
        self.assertEqual(reasoning.detect(HARMONY, "gpt-oss-20b-F16").levels,
                         ("low", "medium", "high"))

    def test_a_preset_level_is_one_of_these_names(self):
        # The gpt-oss presets carry {"reasoning": "medium"}, and app.py
        # looks that string up in spec.levels / spec.kwargs.
        self.assertIn("medium", self.spec().kwargs)


class BothHalvesAreNeeded(unittest.TestCase):
    """Each of the two causes, held on its own."""

    def test_without_a_documented_list_there_is_no_level(self):
        # An unknown model whose template reads reasoning_effort without
        # listing values: still a reported problem, never a guessed list.
        spec = reasoning.detect(HARMONY, "some-unknown-20b")
        self.assertEqual(spec.levels, ())
        self.assertIn("does not list the values it accepts", spec.problem)
        self.assertTrue(spec.thinks)

    def test_without_the_statement_default_there_is_no_level(self):
        # The same template with its default written nowhere detect() can
        # read: a documented list is not enough on its own.
        blind = HARMONY.replace('{%- set reasoning_effort = "medium" %}',
                                '{%- set reasoning_effort = "" %}')
        spec = reasoning.detect(blind, GPTOSS_ID)
        self.assertEqual(spec.levels, ())
        self.assertIn("does not say which one it uses by default",
                      spec.problem)

    def test_the_statement_form_is_read_as_a_default(self):
        self.assertEqual(
            reasoning._default_value(HARMONY, "reasoning_effort"), "medium")

    def test_the_two_older_forms_still_read(self):
        self.assertEqual(
            reasoning._default_value("{{ x|default('xhigh') }}", "x"), "xhigh")
        self.assertEqual(
            reasoning._default_value(
                "{%- set y = x if x is defined else 'low' %}", "x"), "low")

    def test_a_default_set_for_a_different_variable_is_not_borrowed(self):
        # model_identity is defaulted the same way two lines earlier.
        self.assertEqual(
            reasoning._default_value(HARMONY, "reasoning_strength"), "")


class OtherModelsAreUnmoved(unittest.TestCase):
    """The fixtures the gpt-oss fix must not disturb."""

    def test_qwen38_keeps_its_switch_and_three_efforts(self):
        spec = reasoning.detect(QWEN38, "Qwen3.8-27B-UD-Q4_K_XL")
        self.assertEqual(spec.levels, ("off", "low", "medium", "xhigh"))
        self.assertEqual(spec.default, "xhigh")
        self.assertEqual(spec.off, "off")
        self.assertEqual(spec.kwargs["medium"],
                         {"enable_thinking": True,
                          "reasoning_effort": "medium"})

    def test_ornith_is_still_off_or_on(self):
        spec = reasoning.detect(ORNITH, "Ornith-1.5-9B-Q4_K_M")
        self.assertEqual(spec.levels, ("off", "on"))
        self.assertEqual(spec.default, "on")
        self.assertEqual(spec.kwargs,
                         {"off": {"enable_thinking": False},
                          "on": {"enable_thinking": True}})

    def test_granite_still_offers_low_effort(self):
        spec = reasoning.detect(GRANITE, "granite4.2:30b")
        self.assertEqual(spec.levels, ("off", "low-effort", "on"))
        self.assertEqual(spec.kwargs["low-effort"],
                         {"enable_thinking": True, "low_effort": True})
        self.assertEqual(spec.kwargs["on"],
                         {"enable_thinking": True, "low_effort": False})

    def test_muse_glimmer_still_has_four_strengths_and_no_off(self):
        spec = reasoning.detect(MUSE, "muse-glimmer:12b")
        self.assertEqual(spec.levels, ("low", "medium", "high", "xhigh"))
        self.assertEqual(spec.default, "medium")
        self.assertEqual(spec.off, "")

    def test_a_template_with_nothing_to_switch_stays_empty(self):
        spec = reasoning.detect("{{- '<|im_start|>assistant\\n' }}",
                                "Ministral-3-14B-Instruct-2512-Q5_K_M")
        self.assertEqual(spec.levels, ())
        self.assertFalse(spec.thinks)
        self.assertEqual(spec.problem, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
