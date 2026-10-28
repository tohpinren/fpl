import copy
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from forecasting import (
    FEATURE_NAMES,
    fit_model,
    make_features,
    make_samples,
    prediction_cutoff,
    training_before,
)
from historical_data import validate_history


def season_fixture():
    fixtures, events, history = [], [], []
    opening = datetime(2025, 8, 1, 15, tzinfo=timezone.utc)
    for gw in range(1, 11):
        kickoff = opening + timedelta(days=(gw - 1) * 7)
        fixtures.append(
            {
                "id": gw,
                "event": gw,
                "kickoff_time": kickoff.isoformat(),
                "finished": True,
                "team_h": 1,
                "team_a": 2,
                "team_h_score": 1,
                "team_a_score": 0,
            }
        )
        events.append(
            {
                "id": gw,
                "finished": True,
                "deadline_time": (kickoff - timedelta(minutes=90)).isoformat(),
            }
        )
        history.append(
            {
                "element": 1,
                "fixture": gw,
                "round": gw,
                "kickoff_time": kickoff.isoformat(),
                "was_home": True,
                "minutes": 90,
                "starts": 1,
                "total_points": 5,
                "expected_goals": 0.1,
                "expected_assists": 0.2,
                "value": 50,
            }
        )
    player = {"id": 1, "element_type": 2, "team": 1, "total_points": 50, "now_cost": 50}
    bootstrap = {"elements": [player], "teams": [{"id": 1}, {"id": 2}], "events": events}
    return bootstrap, fixtures, {1: history}


class PointInTimeTests(unittest.TestCase):
    def setUp(self):
        self.bootstrap, self.fixtures, self.histories = season_fixture()
        self.cutoff = prediction_cutoff(self.bootstrap, self.fixtures, 5)

    def features(self, bootstrap=None, fixtures=None, history=None, horizon=5):
        return make_features(
            (bootstrap or self.bootstrap)["elements"][0],
            history or self.histories[1],
            fixtures or self.fixtures,
            5,
            horizon,
            self.cutoff,
        )

    def test_features_ignore_future_outcomes_and_final_bootstrap_totals(self):
        expected = self.features()["x"]
        bootstrap, fixtures, history = copy.deepcopy(
            (self.bootstrap, self.fixtures, self.histories[1])
        )
        bootstrap["elements"][0].update(
            team=2, total_points=999, now_cost=999, form=999, ep_next=999
        )
        for row in history[4:]:
            row.update(total_points=999, minutes=1, expected_goals=999, value=999)
        for fixture in fixtures[4:]:
            fixture.update(team_h_score=99, team_a_score=99, team_h_difficulty=5)
        np.testing.assert_allclose(expected, self.features(bootstrap, fixtures, history)["x"])
        self.assertEqual(len(expected), len(FEATURE_NAMES))

    def test_old_round_played_after_cutoff_is_excluded(self):
        history = copy.deepcopy(self.histories[1])
        postponed = {
            **history[0],
            "round": 1,
            "kickoff_time": self.fixtures[7]["kickoff_time"],
            "total_points": 1000,
        }
        np.testing.assert_allclose(
            self.features()["x"], self.features(history=history + [postponed])["x"]
        )

    def test_zero_minute_appearances_affect_minutes_and_recent_form(self):
        history = copy.deepcopy(self.histories[1])
        history[3].update(minutes=0, total_points=0, starts=0)
        values = dict(zip(FEATURE_NAMES, self.features(history=history)["x"]))
        self.assertEqual(values["recent_minutes"], 67.5)
        self.assertEqual(values["appearance_rate"], 0.75)
        self.assertEqual(values["recent_ppg_5"], 3.75)

    def test_only_complete_horizons_become_training_targets(self):
        samples = make_samples(self.bootstrap, self.fixtures, self.histories, horizon=5)
        self.assertEqual({row["gw"] for row in samples}, {4, 5, 6})
        self.assertTrue(all(row["target"] == 25 for row in samples))
        self.bootstrap["events"][-1]["finished"] = False
        samples = make_samples(self.bootstrap, self.fixtures, self.histories, horizon=5)
        self.assertEqual({row["gw"] for row in samples}, {4, 5})

    def test_missing_future_appearances_score_zero_without_dropping_player(self):
        self.histories[1] = self.histories[1][:4]
        sample = next(
            row
            for row in make_samples(self.bootstrap, self.fixtures, self.histories, horizon=5)
            if row["gw"] == 5
        )
        self.assertEqual(sample["target"], 0)

    def test_training_requires_mature_disjoint_labels(self):
        rows = [
            {"gw": 5, "label_end": 100},
            {"gw": 6, "label_end": 100},
            {"gw": 4, "label_end": 300},
        ]
        self.assertEqual(training_before(rows, origin=10, cutoff=200, horizon=5), [rows[0]])

    def test_shuffling_fixtures_does_not_change_features(self):
        np.testing.assert_allclose(
            self.features()["x"], self.features(fixtures=list(reversed(self.fixtures)))["x"]
        )

    def test_blank_horizon_predicts_zero(self):
        features = self.features(fixtures=self.fixtures[:4])
        for name in ("recent_baseline", "minutes_baseline"):
            self.assertEqual(fit_model(name, []).predict([features])[0], 0)

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            make_samples(self.bootstrap, self.fixtures, self.histories, horizon=0)

    def test_ridge_preprocessing_is_fitted_only_to_training_rows(self):
        samples = make_samples(self.bootstrap, self.fixtures, self.histories, horizon=5)
        model = fit_model("ridge", samples)
        np.testing.assert_allclose(
            model.model[0].mean_, np.mean([row["x"] for row in samples], axis=0)
        )


class HistoryValidationTests(unittest.TestCase):
    def test_incomplete_season_is_rejected(self):
        bootstrap, fixtures, histories = season_fixture()
        with self.assertRaisesRegex(ValueError, "380"):
            validate_history(bootstrap, fixtures, histories)


if __name__ == "__main__":
    unittest.main()
