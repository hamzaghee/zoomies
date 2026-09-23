r"""
Zoomies - the rules that learn what a card will really hand out.

    python -m unittest test_vram_limits -v

No model is ever loaded here and %LOCALAPPDATA%\Zoomies is never touched:
the arithmetic in state.note_vram, state.vram_budget and
state.prune_vram_limits is driven directly, against a simulated load and a
fake clock, and the one test that needs a file gets a temporary one.

The load these tests replay is the one that broke the old rule. A model
fills a card over tens of seconds and starts spilling long before it has
finished, so a spill is mostly made of readings of a card that is still
filling up. The old rule kept the *lowest* of them and pinned two 16 GB
cards at 7.34 and 7.89 GB; the numbers below are the ones that were in
vram_limits.json when that was found, kept as fixtures so the same mistake
cannot come back quietly.
"""

import json
import os
import shutil
import tempfile
import time
import unittest

import state

GB = 1024 ** 3
STICKER = 17130815488            # RX 6800 XT, as DXGI reports it: 15.95 GiB
LUID = "00000000_0001231F"

# What the old rule wrote for the two cards in this machine, and what a
# believable reading off the same hardware looks like.
PINNED_LOW = 7885209600          # 7.34 GiB
PINNED_LOW_2 = 8468541440        # 7.89 GiB
BELIEVABLE = 15160070144         # 14.12 GiB

# One spill, poll by poll: dedicated usage climbing from the 2 GB the spill
# detector needs up to 13.6 GB, and the overflow still growing after the
# card itself has stopped taking anything. Then the model sits there.
RAMP = [(2.0, 0.8), (4.5, 0.9), (7.3, 1.0), (10.1, 1.2), (12.4, 1.6),
        (13.6, 2.1), (13.6, 2.9), (13.6, 3.4)]
PLATEAU = [(13.6, 3.4)] * 12
PEAK = int(13.6 * GB)


class Clock:
    """A clock the tests move by hand, so an eight second settle window and
    a thirty day expiry do not cost eight seconds and thirty days."""

    def __init__(self, start=1_700_000_000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


class FrozenTime(unittest.TestCase):
    """Base for the tests that need the clock under their control."""

    def setUp(self):
        self.clock = Clock()
        self._real_time = time.time
        time.time = self.clock
        self.addCleanup(self.restore)

    def restore(self):
        time.time = self._real_time


def replay(samples, cards=None, watch=None, clock=None, poll=0.2, every=2.0):
    """Offer a load to note_vram the way the app does: the counters are read
    every `every` seconds and the refresh loop hands each reading on every
    `poll`. Returns (cards, the indexes of the samples that taught it
    something)."""
    cards = {} if cards is None else cards
    learned = []
    for i, (used, spilled) in enumerate(samples):
        for _ in range(int(round(every / poll))):
            if state.note_vram(cards, LUID, STICKER, int(used * GB),
                               int(spilled * GB), watch):
                learned.append(i)
            if clock is not None:
                clock.tick(poll)
    return cards, learned


class TheMeasurement(FrozenTime):
    """What a spill is worth as evidence."""

    def test_the_peak_of_a_spill_is_learned_not_the_start_of_it(self):
        cards, _ = replay(RAMP, clock=self.clock)
        self.assertEqual(cards[LUID]["ceiling"], PEAK)

    def test_the_first_spilling_reading_is_not_the_answer(self):
        # The regression proper: the old rule kept this one.
        cards, _ = replay(RAMP, clock=self.clock)
        self.assertGreater(cards[LUID]["ceiling"], int(RAMP[0][0] * GB))

    def test_a_smaller_later_spill_does_not_lower_the_ceiling(self):
        cards, _ = replay(RAMP, clock=self.clock)
        replay([(9.0, 1.0)] * 4, cards=cards, clock=self.clock)
        self.assertEqual(cards[LUID]["ceiling"], PEAK)

    def test_a_spill_never_lowers_a_floor_a_clean_run_proved(self):
        cards = {LUID: {"vram": STICKER, "clean": int(12.4 * GB)}}
        replay([(11.0, 1.0)], cards=cards, clock=self.clock)
        self.assertEqual(cards[LUID]["clean"], int(12.4 * GB))

    def test_a_clean_run_keeps_the_highest_it_reached(self):
        cards, _ = replay([(8.0, 0), (13.9, 0), (4.0, 0)], clock=self.clock)
        self.assertEqual(cards[LUID]["clean"], int(13.9 * GB))

    def test_a_clean_run_alone_never_invents_a_ceiling(self):
        cards, _ = replay([(13.9, 0)], clock=self.clock)
        self.assertIsNone(cards[LUID].get("ceiling"))
        self.assertEqual(state.vram_budget(cards, LUID, STICKER), STICKER)

    def test_nothing_is_learned_from_a_card_with_no_reading(self):
        cards = {}
        self.assertFalse(state.note_vram(cards, LUID, STICKER, 0, 0))
        self.assertFalse(state.note_vram(cards, "", STICKER, 1 * GB, 0))
        self.assertFalse(state.note_vram(cards, LUID, 0, 1 * GB, 0))
        self.assertEqual(cards, {})


class Settling(FrozenTime):
    """Only a card that has stopped growing is measuring anything."""

    def test_nothing_is_learned_while_the_footprint_is_still_growing(self):
        watch = {}
        _cards, learned = replay(RAMP + PLATEAU, watch=watch, clock=self.clock)
        self.assertTrue(learned, "the settled load taught nothing at all")
        self.assertGreaterEqual(min(learned), len(RAMP))

    def test_the_settled_load_is_learned_at_the_same_peak(self):
        cards, _ = replay(RAMP + PLATEAU, watch={}, clock=self.clock)
        self.assertEqual(cards[LUID]["ceiling"], PEAK)

    def test_dedicated_usage_going_flat_is_not_enough(self):
        # A card that has run out stops taking dedicated VRAM while the
        # model keeps loading into system RAM. Watching "used" alone would
        # call this settled; watching the whole footprint does not.
        filling = [(13.6, 2.1), (13.6, 2.9), (13.6, 3.7), (13.6, 4.5)]
        _cards, learned = replay(filling, watch={}, clock=self.clock)
        self.assertEqual(learned, [])

    def test_a_load_creeping_up_a_few_mb_at_a_time_is_still_growing(self):
        creep = [(13.0 + n * 0.03, 1.0) for n in range(20)]
        _cards, learned = replay(creep, watch={}, clock=self.clock)
        self.assertEqual(learned, [])

    def test_without_a_watch_every_reading_counts(self):
        _cards, learned = replay(RAMP, watch=None, clock=self.clock)
        self.assertTrue(learned)


class WritingItDown(FrozenTime):
    """A spill already on record is not news."""

    def test_a_settled_spill_is_only_written_down_once(self):
        _cards, learned = replay(RAMP + PLATEAU, watch={}, clock=self.clock)
        self.assertEqual(len(learned), 1)

    def test_seen_counts_confirmations_rather_than_readings(self):
        cards, _ = replay(RAMP + PLATEAU, watch={}, clock=self.clock)
        self.assertEqual(cards[LUID]["seen"], 1)
        self.clock.tick(state.VRAM_CONFIRM_SECONDS + 1)
        replay(PLATEAU, cards=cards, watch={}, clock=self.clock)
        self.assertEqual(cards[LUID]["seen"], 2)

    def test_a_higher_peak_is_written_down_at_once(self):
        cards, _ = replay(RAMP + PLATEAU, watch={}, clock=self.clock)
        replay([(14.8, 3.4)] * 12, cards=cards, watch={}, clock=self.clock)
        self.assertEqual(cards[LUID]["ceiling"], int(14.8 * GB))


class Credible(unittest.TestCase):
    """What cannot be a real reserve is not believed, however it got in."""

    def test_the_readings_that_pinned_this_machines_cards_are_refused(self):
        for low in (PINNED_LOW, PINNED_LOW_2):
            cards = {}
            state.note_vram(cards, LUID, STICKER, low, 1 * GB)
            self.assertIsNone(cards[LUID].get("ceiling"),
                              "%d was believed" % low)

    def test_a_believable_reading_off_the_same_card_is_kept(self):
        cards = {}
        state.note_vram(cards, LUID, STICKER, BELIEVABLE, 1 * GB)
        self.assertEqual(cards[LUID]["ceiling"], BELIEVABLE)

    def test_an_idle_desktop_plus_windows_own_reserve_still_fits(self):
        # This machine idles at up to 2.4 GB of dedicated VRAM and Windows
        # holds back more on top. A ceiling that low down must survive.
        cards = {}
        state.note_vram(cards, LUID, STICKER, int(12.5 * GB), 1 * GB)
        self.assertEqual(cards[LUID]["ceiling"], int(12.5 * GB))

    def test_a_low_ceiling_in_the_file_is_ignored_even_unpruned(self):
        cards = {LUID: {"vram": STICKER, "ceiling": PINNED_LOW}}
        self.assertEqual(state.vram_budget(cards, LUID, STICKER), STICKER)

    def test_the_reserve_is_capped_on_a_big_card(self):
        for size in (16, 24, 48):
            vram = size * GB
            self.assertEqual(vram - state.least_credible_ceiling(vram),
                             state.VRAM_MAX_RESERVE)

    def test_a_small_card_falls_back_to_the_share(self):
        # Subtracting a fixed 5 GB from a 4 GB card would believe anything.
        self.assertEqual(state.least_credible_ceiling(4 * GB),
                         int(4 * GB * state.VRAM_MIN_CEILING))


class Budget(unittest.TestCase):
    """What the estimate is handed."""

    def test_an_unmeasured_card_is_worth_its_sticker_vram(self):
        self.assertEqual(state.vram_budget({}, LUID, STICKER), STICKER)

    def test_a_ceiling_holds_the_card_below_its_sticker_vram(self):
        cards = {LUID: {"vram": STICKER, "ceiling": int(13.6 * GB)}}
        self.assertEqual(state.vram_budget(cards, LUID, STICKER),
                         int(13.6 * GB))

    def test_a_clean_run_above_the_ceiling_says_the_ceiling_has_moved(self):
        cards = {LUID: {"vram": STICKER, "ceiling": int(13.6 * GB),
                        "clean": int(14.2 * GB)}}
        self.assertEqual(state.vram_budget(cards, LUID, STICKER),
                         int(14.2 * GB))

    def test_the_budget_never_exceeds_the_card(self):
        cards = {LUID: {"vram": STICKER, "ceiling": STICKER,
                        "clean": STICKER * 2}}
        self.assertEqual(state.vram_budget(cards, LUID, STICKER), STICKER)


class Expiry(FrozenTime):
    """A ceiling is re-earned rather than binding forever."""

    def aged(self, days):
        return {LUID: {"vram": STICKER, "ceiling": BELIEVABLE,
                       "clean": int(13.0 * GB), "seen": 40,
                       "ts": int(self.clock() - days * 86400)}}

    def test_a_ceiling_nothing_has_confirmed_for_a_month_lapses(self):
        cards = self.aged(state.VRAM_CEILING_DAYS + 10)
        self.assertTrue(state.prune_vram_limits(cards))
        self.assertIsNone(cards[LUID].get("ceiling"))

    def test_the_floor_it_was_measured_beside_survives(self):
        cards = self.aged(state.VRAM_CEILING_DAYS + 10)
        state.prune_vram_limits(cards)
        self.assertEqual(cards[LUID]["clean"], int(13.0 * GB))

    def test_a_ceiling_confirmed_this_week_is_kept(self):
        cards = self.aged(3)
        self.assertFalse(state.prune_vram_limits(cards))
        self.assertEqual(cards[LUID]["ceiling"], BELIEVABLE)

    def test_an_old_entry_is_dated_by_the_stamp_it_does_have(self):
        # Entries written before "ts" existed carry only "at".
        stale = time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(self._real_time()
                                             - 400 * 86400))
        cards = {LUID: {"vram": STICKER, "ceiling": BELIEVABLE, "at": stale}}
        self.restore()          # strptime/mktime want the real calendar
        self.assertTrue(state.prune_vram_limits(cards))

    def test_an_entry_with_no_date_at_all_is_left_alone(self):
        cards = {LUID: {"vram": STICKER, "ceiling": BELIEVABLE}}
        self.assertFalse(state.prune_vram_limits(cards))


class Pruning(unittest.TestCase):
    """Repairing the file, once, on the way in."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "vram_limits.json")
        self._real_path = state.VRAM_LIMITS_PATH
        state.VRAM_LIMITS_PATH = self.path
        self.addCleanup(setattr, state, "VRAM_LIMITS_PATH", self._real_path)
        self.write({
            "00000000_0001231F": {"vram": STICKER, "clean": 12402786304,
                                  "ceiling": PINNED_LOW, "seen": 1064,
                                  "at": "2026-09-23 15:11"},
            "00000000_00011FF2": {"vram": STICKER, "clean": 15391420416,
                                  "ceiling": BELIEVABLE, "seen": 90,
                                  "ts": int(time.time())},
        })

    def write(self, cards):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "cards": cards}, fh)

    def read(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)["cards"]

    def test_the_file_is_repaired_on_the_way_in(self):
        cards = state.load_vram_limits()
        self.assertIsNone(cards["00000000_0001231F"].get("ceiling"))
        self.assertEqual(cards["00000000_00011FF2"]["ceiling"], BELIEVABLE)

    def test_the_repair_is_written_back(self):
        state.load_vram_limits()
        self.assertIsNone(self.read()["00000000_0001231F"].get("ceiling"))

    def test_what_is_returned_is_what_was_written(self):
        self.assertEqual(state.load_vram_limits(), self.read())

    def test_the_seen_count_goes_with_the_ceiling_it_counted(self):
        cards = state.load_vram_limits()
        self.assertIsNone(cards["00000000_0001231F"].get("seen"))

    def test_the_floor_is_never_dropped(self):
        cards = state.load_vram_limits()
        self.assertEqual(cards["00000000_0001231F"]["clean"], 12402786304)

    def test_pruning_again_changes_nothing(self):
        self.assertFalse(state.prune_vram_limits(state.load_vram_limits()))

    def test_a_pinned_card_goes_back_to_its_sticker_vram(self):
        # 11.55 GiB under the old rule, because the bad ceiling dragged the
        # clean floor down with it. 15.95 GiB now, until a spill says less.
        cards = state.load_vram_limits()
        self.assertEqual(
            state.vram_budget(cards, "00000000_0001231F", STICKER), STICKER)

    def test_rubbish_in_the_file_costs_the_entry_not_the_app(self):
        self.write({LUID: "not a card"})
        self.assertEqual(state.load_vram_limits(), {})

    def test_a_missing_file_is_not_an_error(self):
        os.remove(self.path)
        self.assertEqual(state.load_vram_limits(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
