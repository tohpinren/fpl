import json
import subprocess
import tempfile
import unittest
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch

import serve


class DashboardHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(serve.DashboardHandler, directory=str(serve.FRONTEND_DIR)),
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, headers=None):
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        self.addCleanup(connection.close)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read(), response.headers

    def test_frontend_and_api_are_served(self):
        status, body, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Touchline", body)
        with patch.object(serve, "build_dashboard", return_value={"players": []}):
            status, body, headers = self.request("GET", "/api/dashboard")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"players": []})
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_repository_files_are_not_exposed(self):
        for path in ("/serve.py", "/data/bootstrap-static.json", "/../serve.py"):
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 404)

    def test_missing_data_has_an_actionable_error(self):
        with patch.object(serve, "build_dashboard", side_effect=FileNotFoundError):
            status, body, _ = self.request("GET", "/api/dashboard")
        self.assertEqual(status, 503)
        self.assertIn("Refresh data", json.loads(body)["error"])

    def test_refresh_runs_existing_downloader_and_returns_updated_data(self):
        result = Mock(returncode=0, stdout="Download complete!", stderr="")
        with patch.object(serve.subprocess, "run", return_value=result) as run, patch.object(
            serve, "build_dashboard", return_value={"season": "2026/27"}
        ):
            status, body, _ = self.request("POST", "/api/refresh")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["season"], "2026/27")
        self.assertEqual(run.call_args.args[0][1], str(serve.ROOT / "download_fpl_data.py"))
        self.assertEqual(run.call_args.kwargs["cwd"], serve.ROOT)

    def test_partial_refresh_reports_failure_and_releases_lock(self):
        result = Mock(returncode=0, stdout="Error fetching /element-summary/1/", stderr="")
        with patch.object(serve.subprocess, "run", return_value=result):
            status, _, _ = self.request("POST", "/api/refresh")
        self.assertEqual(status, 502)
        self.assertFalse(serve.REFRESH_LOCK.locked())

    def test_timeout_reports_failure_and_releases_lock(self):
        with patch.object(
            serve.subprocess, "run", side_effect=subprocess.TimeoutExpired("download", 180)
        ):
            status, _, _ = self.request("POST", "/api/refresh")
        self.assertEqual(status, 504)
        self.assertFalse(serve.REFRESH_LOCK.locked())

    def test_overlapping_refresh_is_rejected(self):
        with serve.REFRESH_LOCK:
            status, _, _ = self.request("POST", "/api/refresh")
            read_status, _, _ = self.request("GET", "/api/dashboard")
        self.assertEqual(status, 409)
        self.assertEqual(read_status, 409)

    def test_other_websites_cannot_trigger_refresh(self):
        with patch.object(serve.subprocess, "run") as run:
            status, _, _ = self.request("POST", "/api/refresh", {"Origin": "https://example.com"})
        self.assertEqual(status, 403)
        run.assert_not_called()


class DashboardDataTests(unittest.TestCase):
    def test_no_local_data_fails_without_creating_fake_players(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(serve, "DATA_DIR", Path(folder)):
            with self.assertRaises(FileNotFoundError):
                serve.build_dashboard()


if __name__ == "__main__":
    unittest.main()
