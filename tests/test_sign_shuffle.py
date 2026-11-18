"""Check sign randomization and its integration with the shared comparison."""

import random
import sys
import unittest
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pattern_counts import brute_force_counts
from sign_shuffle import shuffle_signs
from time_shuffle import compare


class SignShuffleTests(unittest.TestCase):
    def setUp(self):
        t = datetime(2020, 1, 1)
        self.events = [("a", "b", t, -1), ("a", "b", t, 1),
                       ("b", "a", t + timedelta(days=1), -1),
                       ("c", "b", t + timedelta(days=2), 1)]

    def test_all_negative_preserves_parallel_events(self):
        self.assertCountEqual(shuffle_signs(self.events, 4, random.Random(0)),
                              [e[:3] for e in self.events])

    def test_sample_size_and_reproducibility(self):
        first = shuffle_signs(self.events, 2, random.Random(4))
        self.assertEqual(first, shuffle_signs(self.events, 2, random.Random(4)))
        self.assertEqual(len(first), 2)
        self.assertTrue(all(event in [e[:3] for e in self.events] for event in first))

    def test_zero_negative_events(self):
        self.assertEqual(shuffle_signs(self.events, 0, random.Random(0)), [])

    def test_comparison_matches_independent_counter(self):
        negative = [e[:3] for e in self.events if e[3] == -1]
        sampler = partial(shuffle_signs, self.events, 2)
        rows = list(compare(negative, [1, 3], 3, 0, sampler=sampler))
        observed = brute_force_counts(negative, [1, 3])
        rng = random.Random(0)
        expected = [brute_force_counts(sampler(rng), [1, 3]) for _ in range(3)]
        for delta, pattern, metric, actual, center, *_ in rows:
            self.assertEqual(actual, observed[delta][pattern][metric])
            self.assertAlmostEqual(center, sum(r[delta][pattern][metric] for r in expected) / 3)

    def test_degenerate_null_has_blank_z_score(self):
        negative = [e[:3] for e in self.events]
        rows = list(compare(negative, [1], 2, 0,
                            sampler=partial(shuffle_signs, self.events, 4)))
        self.assertTrue(all(row[5] == 0 and row[6] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
