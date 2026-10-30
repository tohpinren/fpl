import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import build_site


class StaticBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        (self.data / "players").mkdir(parents=True)
        (self.data / "bootstrap-static.json").write_text(json.dumps({"elements": [{"id": 1}]}))
        (self.data / "players/1.json").write_text('{"history": []}')
        patcher = patch.object(build_site, "DATA_DIR", self.data)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.output = self.root / "dist"

    def test_export_contains_only_public_assets_and_snapshot(self):
        snapshot = {"season": "2026/27", "players": [{"name": "Ødegaard"}]}
        with patch.object(build_site, "build_dashboard", return_value=snapshot):
            build_site.build_site(self.output)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"index.html", "app.js", "styles.css", "favicon.svg", "dashboard.json", ".nojekyll"},
        )
        self.assertEqual(json.loads((self.output / "dashboard.json").read_text()), snapshot)
        html = (self.output / "index.html").read_text()
        self.assertIn('name="dashboard-mode" content="static"', html)
        self.assertNotIn('href="/', html)
        self.assertNotIn('src="/', html)
        self.assertFalse((self.output / "data").exists())
        self.assertFalse((self.output / "serve.py").exists())

    def test_partial_download_does_not_replace_previous_build(self):
        self.output.mkdir()
        published = self.output / "dashboard.json"
        published.write_text('{"previous": true}')
        result = Mock(stdout="Error fetching /element-summary/1/", returncode=0)
        with patch.object(build_site.subprocess, "run", return_value=result), self.assertRaises(
            RuntimeError
        ):
            build_site.build_site(self.output, refresh=True)
        self.assertEqual(json.loads(published.read_text()), {"previous": True})

    def test_missing_player_history_blocks_export(self):
        (self.data / "players/1.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing 1 player"):
            build_site.build_site(self.output)
        self.assertFalse(self.output.exists())

    def test_failed_download_aborts_build(self):
        with patch.object(
            build_site.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "download")
        ), self.assertRaises(subprocess.CalledProcessError):
            build_site.build_site(self.output, refresh=True)
        self.assertFalse(self.output.exists())

    def test_history_is_restored_before_forecasting_and_saved_before_export(self):
        self.data.joinpath("fixtures.json").write_text("[]")
        snapshot = {"model": {}, "players": []}
        report = dict(pending_origins=[9], completed_windows=[])
        with patch("prospective.github_ledger") as factory, patch("prospective.capture") as capture, \
                patch("prospective.evaluate", return_value=report), \
                patch("download_fpl_data.get_season", return_value="2026-27"), \
                patch("forecasting.load_live_histories", return_value={}), \
                patch.object(build_site, "build_dashboard", return_value=snapshot) as dashboard:
            parent = Mock()
            parent.attach_mock(factory.return_value.restore, "restore")
            parent.attach_mock(dashboard, "dashboard")
            parent.attach_mock(capture, "capture")
            parent.attach_mock(factory.return_value.publish, "publish")
            build_site.build_site(self.output, forecast_repository="owner/repo")
        self.assertEqual([c[0] for c in parent.mock_calls], ["restore", "dashboard", "capture", "publish"])
        result = json.loads((self.output / "dashboard.json").read_text())
        self.assertEqual(result["model"]["prospective_test"]["pending_origins"], [9])
        self.assertFalse((self.output / "forecast-ledger").exists())


if __name__ == "__main__":
    unittest.main()
