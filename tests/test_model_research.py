"""Guard the research experiment against temporal and scoring-rule leakage."""

import copy
import unittest

from forecasting import make_samples
from research.model_improvements import context, extras, fixture_samples, fixture_weights, stable_points
from test_forecasting import season_fixture


class ResearchValidityTests(unittest.TestCase):
    def test_context_excludes_future_stale_and_wrong_origin_snapshots(self):
        sample = dict(id=1, gw=10, cutoff=1_000_000)
        player = dict(id=1, status="i", chance_of_playing_next_round=0)
        unavailable = [dict(timestamp=1_000_001, players={1: player}),
                       dict(timestamp=1, players={1: player}),
                       dict(timestamp=999_999, origin=11, players={1: player})]
        for snapshot in unavailable:
            self.assertEqual(context(sample, [snapshot]), [0] * 9)
        fresh = dict(timestamp=999_999, players={1: player})
        result = context(sample, [fresh])
        self.assertEqual(result[0], 1)
        self.assertEqual(result[4:7], [1, 1, 0])

    def test_missing_availability_probability_is_not_perfect_fitness(self):
        sample = dict(id=1, gw=10, cutoff=1_000_000)
        snapshot = dict(timestamp=999_999, players={1: dict(status="a")})
        result = context(sample, [snapshot])
        self.assertEqual(result[2], 1)
        self.assertEqual(result[5:7], [0, 0])

    def test_extra_features_ignore_future_match_statistics(self):
        b, f, h = season_fixture()
        samples = [s for s in make_samples(b, f, h) if s["gw"] == 5]
        before = extras(b, f, h, samples)[0]["extra"]
        changed = copy.deepcopy(h)
        for row in changed[1][4:]:
            row.update(expected_goals=999, minutes=1, defensive_contribution=999)
        self.assertEqual(before, extras(b, f, changed, samples)[0]["extra"])

    def test_stable_subtotal_excludes_rule_sensitive_points(self):
        row = dict(minutes=90, goals_scored=1, clean_sheets=1, goals_conceded=0,
                   assists=1, bonus=3, defensive_contribution=20, total_points=999)
        self.assertEqual(stable_points(row, "DEF"), 12)
        self.assertEqual(stable_points({**row, "minutes": 0, "goals_scored": 0, "clean_sheets": 0}, "DEF"), 0)

    def test_fixture_labels_include_missing_appearances_and_ignore_future_in_x(self):
        b, f, h = season_fixture()
        h[1] = h[1][:4]
        rows = fixture_samples(b, f, h, "test")
        sample = next(r for r in rows if r["gw"] == 5)
        self.assertEqual((sample["minutes_class"], sample["target"]), (0, 0))
        self.assertEqual(len({(r["id"], r["gw"], r["fixture"]) for r in rows}), len(rows))
        # Training sees the same lead times that a five-GW forecast will use.
        self.assertTrue(any(r["gw"] == 4 and r["fixture"] == 8 for r in rows))
        weights = fixture_weights(rows)
        self.assertAlmostEqual(sum(w for r, w in zip(rows, weights) if r["fixture"] == 8), 1)
        h[1].append(dict(h[1][-1], fixture=5, round=5,
                         kickoff_time=f[4]["kickoff_time"], total_points=999, minutes=10))
        changed = next(r for r in fixture_samples(b, f, h, "test") if r["gw"] == 5)
        self.assertEqual(changed["x"], sample["x"])
        self.assertEqual(changed["target"], 999)


if __name__ == "__main__":
    unittest.main()
