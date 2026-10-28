import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import download_fpl_data as downloader
from fpl_scoring import calculate_defcon_stats


def bootstrap(year):
    return {
        "events": [
            {"id": 38, "deadline_time": f"{year + 1}-05-30T13:30:00Z"},
            {"id": 1, "deadline_time": f"{year}-08-21T17:30:00Z"},
        ],
        "elements": [{"id": 1}],
    }


class SeasonDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.data_dir = Path(self.temp_dir.name) / "data"
        patcher = patch.object(downloader, "DATA_DIR", self.data_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_old_season(self):
        downloader.save_json(bootstrap(2025), self.data_dir / "bootstrap-static.json")
        downloader.save_json({"history": ["old player"]}, self.data_dir / "players/1.json")
        downloader.save_json({"history": []}, self.data_dir / "players/999.json")
        downloader.save_json({"elements": []}, self.data_dir / "gameweeks/35.json")

    def test_rollover_preserves_old_season_and_removes_reused_ids(self):
        self.write_old_season()
        with patch.object(downloader, "fetch_endpoint", return_value=bootstrap(2026)):
            downloader.download_bootstrap_static(Mock())
        archive = self.data_dir / "archive/2025-26"
        self.assertEqual(
            json.loads((archive / "players/1.json").read_text())["history"], ["old player"]
        )
        self.assertTrue((archive / "players/999.json").exists())
        self.assertTrue((archive / "gameweeks/35.json").exists())
        self.assertFalse((self.data_dir / "players").exists())
        self.assertFalse((self.data_dir / "gameweeks").exists())
        self.assertEqual(
            downloader.get_season(
                json.loads((self.data_dir / "bootstrap-static.json").read_text())
            ),
            "2026-27",
        )
        # A failed request must never leave another season's history at a reused ID.
        with patch.object(downloader, "fetch_endpoint", return_value=None):
            downloader.download_player_summaries(Mock(), [1])
        self.assertFalse((self.data_dir / "players/1.json").exists())

    def test_same_season_refresh_keeps_history(self):
        self.write_old_season()
        self.assertEqual(downloader.prepare_data_directory(bootstrap(2025)), "2025-26")
        self.assertTrue((self.data_dir / "players/1.json").exists())
        self.assertFalse((self.data_dir / "archive").exists())

    def test_existing_archive_is_never_overwritten(self):
        self.write_old_season()
        (self.data_dir / "archive/2025-26").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            downloader.prepare_data_directory(bootstrap(2026))
        self.assertTrue((self.data_dir / "players/1.json").exists())

    def test_failed_bootstrap_preserves_local_data(self):
        self.write_old_season()
        with patch.object(downloader, "fetch_endpoint", return_value=None):
            self.assertIsNone(downloader.download_bootstrap_static(Mock()))
        self.assertTrue((self.data_dir / "players/1.json").exists())
        self.assertFalse((self.data_dir / "archive").exists())

    def test_only_started_gameweeks_are_requested(self):
        data = bootstrap(2026)
        data["events"] = [
            {"id": 1, "finished": True},
            {"id": 3, "is_current": True},
            {"id": 4, "is_next": True},
        ]
        with patch.object(downloader, "create_session"), patch.object(
            downloader, "download_bootstrap_static", return_value=data
        ), patch.object(downloader, "download_fixtures"), patch.object(
            downloader, "download_player_summaries"
        ), patch.object(downloader, "download_gameweek_data") as download_gw:
            downloader.main()
        self.assertEqual(sorted(call.args[1] for call in download_gw.call_args_list), [1, 3])

    def test_preseason_handles_empty_gameweeks_and_players(self):
        with patch.object(downloader, "ThreadPoolExecutor") as executor:
            downloader.download_all_gameweeks(Mock(), [])
            downloader.download_player_summaries(Mock(), [])
        executor.assert_not_called()

    def test_empty_fixture_list_replaces_stale_fixtures(self):
        downloader.save_json([{"id": 1}], self.data_dir / "fixtures.json")
        with patch.object(downloader, "fetch_endpoint", return_value=[]):
            downloader.download_fixtures(Mock())
        self.assertEqual(json.loads((self.data_dir / "fixtures.json").read_text()), [])


class DefensiveScoringTests(unittest.TestCase):
    def test_points_are_awarded_per_match_with_position_thresholds(self):
        history = [
            {"minutes": 90, "defensive_contribution": value} for value in [9, 10, 11, 12, 24]
        ]
        for position, expected_points, expected_pct in [
            (1, 0, 0),
            (2, 8, 80),
            (3, 4, 40),
            (4, 4, 40),
        ]:
            with self.subTest(position=position):
                stats = calculate_defcon_stats(history, position)
                self.assertEqual(stats["points"], expected_points)
                self.assertEqual(stats["threshold_pct"], expected_pct)

    def test_subthreshold_totals_never_accumulate_into_points(self):
        stats = calculate_defcon_stats([{"minutes": 90, "defensive_contribution": 9}] * 5, 2)
        self.assertEqual(stats["points"], 0)

    def test_no_appearances_or_missing_stats(self):
        for history in [
            [],
            [{"minutes": 0, "defensive_contribution": 20}],
            [{"minutes": 90}],
            [{"minutes": None, "defensive_contribution": None}],
        ]:
            with self.subTest(history=history):
                stats = calculate_defcon_stats(history, 3)
                self.assertEqual(stats["points"], 0)
                self.assertEqual(stats["threshold_pct"], 0)


if __name__ == "__main__":
    unittest.main()
