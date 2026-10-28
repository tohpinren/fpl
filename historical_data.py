"""Download and validate the complete 2025/26 FPL archive reproducibly.

The upstream project archives official FPL API statistics. The revision is pinned
so the model review can be reproduced without silently changing the dataset.
"""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
HISTORY_DIR = ROOT / "data" / "history" / "2025-26"
SOURCE_REPOSITORY = "vaastav/Fantasy-Premier-League"
SOURCE_REVISION = "d7b05b51d137bfefef4b692386d18c610975b9b9"
SOURCE_URL = f"https://github.com/{SOURCE_REPOSITORY}/tree/{SOURCE_REVISION}/data/2025-26"
FILES = ("players_raw.csv", "teams.csv", "fixtures.csv", "gws/merged_gw.csv")


def parse_value(value):
    if value in ("", "None", "null"):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def load_history(directory=HISTORY_DIR, download=True):
    directory = Path(directory)
    manifest_file = directory / "source.json"
    if not manifest_file.exists():
        if not download:
            raise FileNotFoundError(manifest_file)
        directory.mkdir(parents=True, exist_ok=True)
        checksums = {}
        for name in FILES:
            url = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_REVISION}/data/2025-26/{name}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            checksums[name] = hashlib.sha256(response.content).hexdigest()
        manifest_file.write_text(
            json.dumps(
                {"revision": SOURCE_REVISION, "url": SOURCE_URL, "sha256": checksums}, indent=2
            )
        )
    manifest = json.loads(manifest_file.read_text())
    if manifest["revision"] != SOURCE_REVISION:
        raise ValueError("Historical source revision does not match the pinned model dataset.")
    tables = {}
    for name in FILES:
        content = (directory / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Historical file checksum mismatch: {name}")
        tables[name] = [
            {key: parse_value(value) for key, value in row.items()}
            for row in csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        ]
    fixtures = tables["fixtures.csv"]
    for fixture in fixtures:
        fixture.pop("stats", None)
    histories = defaultdict(list)
    unique_rows = {}
    duplicates_removed = 0
    for row in tables["gws/merged_gw.csv"]:
        key = (row["element"], row["fixture"])
        if key in unique_rows:
            if unique_rows[key] != row:
                raise ValueError(f"Conflicting duplicate player-fixture row: {key}")
            duplicates_removed += 1
            continue
        unique_rows[key] = row
        histories[row["element"]].append(row)
    for history in histories.values():
        history.sort(key=lambda row: (row["kickoff_time"], row["fixture"]))
    events = []
    for gw in range(1, 39):
        matches = [fixture for fixture in fixtures if fixture["event"] == gw]
        if not matches:
            raise ValueError(f"Historical archive is missing GW{gw} fixtures.")
        first_kickoff = min(fixture["kickoff_time"] for fixture in matches)
        # Exact pre-deadline fixture snapshots are unavailable. This conservative
        # timestamp proxy is documented in the report; no final FDR is a feature.
        deadline = datetime.fromisoformat(first_kickoff.replace("Z", "+00:00")) - timedelta(
            minutes=90
        )
        events.append(
            {
                "id": gw,
                "deadline_time": deadline.isoformat(),
                "finished": all(f["finished"] for f in matches),
            }
        )
    bootstrap = {
        "elements": tables["players_raw.csv"],
        "teams": tables["teams.csv"],
        "events": events,
    }
    audit = validate_history(bootstrap, fixtures, dict(histories))
    audit["identical_duplicate_rows_removed"] = duplicates_removed
    return bootstrap, fixtures, dict(histories), {**manifest, "audit": audit}


def validate_history(bootstrap, fixtures, histories):
    fixture_ids = {fixture["id"] for fixture in fixtures}
    if len(fixtures) != 380 or len(fixture_ids) != 380:
        raise ValueError("Expected exactly 380 unique Premier League fixtures.")
    if not all(
        f["finished"] and f["team_h_score"] is not None and f["team_a_score"] is not None
        for f in fixtures
    ):
        raise ValueError("The full-season archive contains unfinished fixtures.")
    seen = set()
    rounds = set()
    missing_stats = Counter()
    point_mismatches = []
    player_ids = {player["id"] for player in bootstrap["elements"]}
    for player_id, history in histories.items():
        if player_id not in player_ids:
            raise ValueError(f"Unknown player ID: {player_id}")
        for row in history:
            key = (player_id, row["fixture"])
            if key in seen:
                raise ValueError(f"Duplicate player-fixture row: {key}")
            seen.add(key)
            rounds.add(row["round"])
            if row["fixture"] not in fixture_ids or not 0 <= row["minutes"] <= 120:
                raise ValueError(f"Invalid historical match row: {key}")
            for field in ("expected_goals", "expected_assists", "defensive_contribution"):
                if row.get(field) is None:
                    missing_stats[field] += 1
    for player in bootstrap["elements"]:
        actual = sum(row["total_points"] for row in histories.get(player["id"], []))
        if actual != player["total_points"]:
            point_mismatches.append(
                {
                    "id": player["id"],
                    "bootstrap_points": player["total_points"],
                    "history_points": actual,
                }
            )
    if rounds != set(range(1, 39)):
        raise ValueError("Historical gameweek coverage is incomplete.")
    if missing_stats:
        raise ValueError(f"Historical archive has missing model statistics: {dict(missing_stats)}")
    if point_mismatches:
        raise ValueError(
            f"Historical point totals do not reconcile for {len(point_mismatches)} players."
        )
    return {
        "players": len(player_ids),
        "fixtures": len(fixtures),
        "player_fixture_rows": len(seen),
        "gameweeks": sorted(rounds),
        "duplicate_rows": 0,
        "missing_stats": dict(missing_stats),
        "point_total_mismatches": point_mismatches,
    }


if __name__ == "__main__":
    _, _, _, source = load_history()
    print(json.dumps(source, indent=2))
