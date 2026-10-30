import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from availability import FEATURE_NAMES, add_training_context, context_features
from research.model_improvements import context


class AvailabilityTests(unittest.TestCase):
    def test_live_features_match_the_frozen_research_model(self):
        player = dict(status="d", chance_of_playing_next_round=75, penalties_order=1,
                      corners_and_indirect_freekicks_order=2)
        expected = context(dict(id=1, gw=9, cutoff=1_000_000),
                           [dict(timestamp=999_900, players={1: player})])
        self.assertEqual(context_features(player, 999_900, 1_000_000), expected)
        self.assertEqual(len(expected), len(FEATURE_NAMES))

    def test_missing_and_stale_data_never_means_perfect_fitness(self):
        self.assertEqual(context_features(dict(status="a"), 999_900, 1_000_000)[5:7], [0, 0])
        for observed in (None, 0, 1_000_001):
            self.assertEqual(context_features(dict(status="a"), observed, 1_000_000), [0] * 9)

    def test_future_news_is_not_used(self):
        player = dict(status="i", news_added="2025-01-01T00:00:00Z")
        self.assertEqual(context_features(player, 1, 1000), [0] * 9)

    def test_training_uses_only_archived_context_for_that_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "forecast-ledger/snapshots/2026-27/gw09.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(origin=9, cutoff=950,
                bootstrap_captured_at=900, defender_context={"1": dict(status="i", chance_of_playing_next_round=0)})))
            rows = [dict(position="DEF", id=1, gw=9, cutoff=1000, x=[7]),
                    dict(position="DEF", id=1, gw=8, cutoff=800, x=[7]),
                    dict(position="MID", id=2, gw=9, cutoff=1000, x=[8])]
            result = add_training_context(rows, root, "2026-27")
            self.assertEqual(result[0]["x"][5:8], [1, 1, 0])
            self.assertEqual(result[1]["x"], [7] + [0] * 9)
            self.assertEqual(result[2]["x"], [8])
            self.assertEqual(rows[0]["x"], [7])

    def test_asset_appends_only_defender_features(self):
        rows = [dict(position="DEF", id=1, gw=9, x=[2]), dict(position="MID", id=1, gw=9, x=[3])]
        with patch("availability.historical_context", return_value={"9": {"1": [1] * 9}}):
            result = add_training_context(rows, Path("unused"), "2025-26")
        self.assertEqual(result[0]["x"], [2] + [1] * 9)
        self.assertEqual(result[1]["x"], [3])
