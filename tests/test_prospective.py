import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prospective import GitHubLedger, capture, evaluate, write_json
from test_forecasting import season_fixture


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bootstrap, self.fixtures, self.histories = season_fixture()
        self.record = dict(season="2025-26", origin=5, horizon=5, cutoff=1,
            deadline=self.bootstrap["events"][4]["deadline_time"],
            fixtures=[f for f in self.fixtures if 5 <= f["event"] <= 9],
            predictions=[dict(id=1, position="MID", baseline=20, candidate=25)])
        self.path = self.root / "snapshots/2025-26/gw05.json"
        write_json(self.path, self.record)

    def test_no_scoring_until_every_target_gameweek_is_complete(self):
        self.bootstrap["events"][8]["finished"] = False
        report = evaluate(self.root, self.bootstrap, self.fixtures, self.histories, now=2e9)
        self.assertEqual(report["pending_origins"], [5])
        self.assertEqual(report["completed_windows"], [])

    def test_scoring_waits_for_originally_scheduled_postponed_fixture(self):
        self.fixtures[5].update(event=10, finished=False)
        report = evaluate(self.root, self.bootstrap, self.fixtures, self.histories, now=2e9)
        self.assertEqual(report["completed_windows"], [])

    def test_complete_window_scores_frozen_predictions_and_keeps_zero_absences(self):
        self.histories[1] = self.histories[1][:6]
        report = evaluate(self.root, self.bootstrap, self.fixtures, self.histories, now=2e9)
        result = report["completed_windows"][0]
        self.assertEqual(result["scores"]["baseline"]["mae"], 10)
        self.assertEqual(result["scores"]["candidate"]["mae"], 15)
        self.assertEqual(result["scores"]["candidate"]["top5_points_per_pick"], 10)
        self.assertEqual(report["summary"]["MID"]["origins"], 1)

    def test_unavailable_player_history_is_not_silently_scored_zero(self):
        with self.assertRaisesRegex(ValueError, "missing player histories"):
            evaluate(self.root, self.bootstrap, self.fixtures, {}, now=2e9)

    def test_late_or_changed_forecasts_are_rejected(self):
        bad = {**self.record, "cutoff": 2e9}
        write_json(self.path, bad)
        with self.assertRaisesRegex(ValueError, "Invalid pre-deadline"):
            evaluate(self.root, self.bootstrap, self.fixtures, self.histories, now=2e9)

    def test_capture_skips_late_runs_and_preserves_the_first_prediction(self):
        self.bootstrap["events"][4]["is_next"] = True
        old = self.path.read_bytes()
        with patch("prospective.live_forecasts") as predict:
            self.assertIsNone(capture(self.root, self.root, self.bootstrap, self.fixtures, now=2e9))
            self.assertEqual(capture(self.root, self.root, self.bootstrap, self.fixtures, now=1), self.path)
            predict.assert_not_called()
        self.assertEqual(self.path.read_bytes(), old)


class LedgerPublicationTests(unittest.TestCase):
    def test_changed_snapshot_cannot_be_published_over_an_existing_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "snapshots/2026-27/gw09.json", dict(forecast=10))
            client = GitHubLedger("owner/repo", "unused")
            client.blobs["snapshots/2026-27/gw09.json"] = "different"
            with self.assertRaisesRegex(RuntimeError, "overwrite a frozen forecast"):
                client.publish(root, "2026-27")

    def test_first_history_commit_creates_separate_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "snapshots/2026-27/gw09.json", dict(forecast=10))
            client = GitHubLedger("owner/repo", "unused")
            with patch.object(client, "request", side_effect=[dict(sha="tree"), dict(sha="commit"), {}]) as request:
                self.assertTrue(client.publish(root, "2026-27"))
            self.assertEqual(request.call_args_list[-1].kwargs["json"]["ref"], "refs/heads/codex/forecast-ledger")

    def test_updates_preserve_existing_tree_and_reject_force_push(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "evaluations/2026-27.json", dict(pending_origins=[9]))
            client = GitHubLedger("owner/repo", "unused")
            client.head, client.tree = "head", "base"
            with patch.object(client, "request", side_effect=[dict(sha="tree"), dict(sha="commit"), {}]) as request:
                client.publish(root, "2026-27")
            self.assertEqual(request.call_args_list[0].kwargs["json"]["base_tree"], "base")
            self.assertFalse(request.call_args_list[-1].kwargs["json"]["force"])
