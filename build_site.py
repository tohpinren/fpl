"""Export the dashboard and analysis snapshot for GitHub Pages.

Run `python build_site.py --refresh` to download fresh FPL data before building.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from serve import DATA_DIR, FRONTEND_DIR, ROOT, build_dashboard


def build_site(output_dir: Path, refresh: bool = False, forecast_repository=None) -> None:
    if refresh:
        result = subprocess.run(
            [sys.executable, str(ROOT / "download_fpl_data.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        print(result.stdout, end="")
        if "Error fetching" in result.stdout or "Failed to download" in result.stdout:
            raise RuntimeError("FPL download was incomplete; keeping the existing published site.")

    # A failed player request must not silently produce an incomplete deployment.
    bootstrap = json.loads((DATA_DIR / "bootstrap-static.json").read_text(encoding="utf-8"))
    missing = [
        player["id"]
        for player in bootstrap["elements"]
        if not (DATA_DIR / "players" / f"{player['id']}.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} player summaries. Run the downloader again.")
    github = None
    if forecast_repository:
        from download_fpl_data import get_season
        from prospective import github_ledger

        github = github_ledger(forecast_repository)
        github.restore(DATA_DIR / "forecast-ledger", get_season(bootstrap))
    dashboard = build_dashboard()
    if github:
        from download_fpl_data import get_season
        from forecasting import load_live_histories
        from prospective import LEDGER_BRANCH, capture, evaluate

        fixtures = json.loads((DATA_DIR / "fixtures.json").read_text())
        ledger_dir = DATA_DIR / "forecast-ledger"
        capture(ledger_dir, DATA_DIR, bootstrap, fixtures)
        report = evaluate(ledger_dir, bootstrap, fixtures, load_live_histories(DATA_DIR, bootstrap))
        github.publish(ledger_dir, get_season(bootstrap))
        dashboard["model"]["prospective_test"] = dict(challenger="components_single", position="MID",
            pending_origins=report["pending_origins"],
            completed_origins=[r["origin"] for r in report["completed_windows"] if r["position"] == "MID"],
            history_url=f"https://github.com/{forecast_repository}/tree/{LEDGER_BRANCH}")
    snapshot = json.dumps(dashboard, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Only publish the frontend and its compact data snapshot, never the raw data archive.
    for name in ("app.js", "styles.css", "favicon.svg"):
        shutil.copyfile(FRONTEND_DIR / name, output_dir / name)
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace(
        'name="dashboard-mode" content="local"', 'name="dashboard-mode" content="static"'
    )
    for name in ("app.js", "styles.css", "favicon.svg"):
        version = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()[:12]
        html = html.replace(f'./{name}"', f'./{name}?v={version}"')
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / "dashboard.json").write_text(snapshot, encoding="utf-8")
    (output_dir / ".nojekyll").touch()
    print(f"Static dashboard built: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build the static FPL dashboard")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument(
        "--refresh", action="store_true", help="Download fresh FPL data before building"
    )
    parser.add_argument("--forecast-repository", help="Persist prospective forecasts on the GitHub history branch")
    args = parser.parse_args()
    build_site(args.output, refresh=args.refresh, forecast_repository=args.forecast_repository)


if __name__ == "__main__":
    main()
