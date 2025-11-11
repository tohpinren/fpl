"""
Script to download all available FPL (Fantasy Premier League) API data for the current season.

This script fetches data from the official FPL API endpoints and saves them locally.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://fantasy.premierleague.com/api"
DATA_DIR = Path("data")
RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)


def create_session() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filepath}")


def fetch_endpoint(session: requests.Session, endpoint: str) -> Optional[Dict[str, Any]]:
    """Fetch data from an API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {endpoint}: {e}")
        return None


def download_bootstrap_static(session: requests.Session) -> Optional[Dict[str, Any]]:
    """Download bootstrap-static data containing players, teams, gameweeks, etc."""
    print("Downloading bootstrap-static data...")
    data = fetch_endpoint(session, "/bootstrap-static/")
    if data:
        save_json(data, DATA_DIR / "bootstrap-static.json")
    return data


def download_fixtures(session: requests.Session) -> None:
    """Download all fixtures data."""
    print("Downloading fixtures data...")
    data = fetch_endpoint(session, "/fixtures/")
    if data:
        save_json(data, DATA_DIR / "fixtures.json")


def download_player_summaries(session: requests.Session, player_ids: List[int]) -> None:
    """Download element-summary for each player."""
    print(f"Downloading player summaries for {len(player_ids)} players...")
    players_dir = DATA_DIR / "players"
    players_dir.mkdir(parents=True, exist_ok=True)

    for i, player_id in enumerate(player_ids, 1):
        data = fetch_endpoint(session, f"/element-summary/{player_id}/")
        if data:
            save_json(data, players_dir / f"{player_id}.json")

        if i % 10 == 0:
            print(f"Progress: {i}/{len(player_ids)} players downloaded")
            time.sleep(0.5)
        else:
            time.sleep(0.1)


def download_gameweek_data(session: requests.Session, gameweek: int) -> None:
    """Download data for a specific gameweek."""
    print(f"Downloading gameweek {gameweek} data...")
    data = fetch_endpoint(session, f"/event/{gameweek}/live/")
    if data:
        save_json(data, DATA_DIR / "gameweeks" / f"{gameweek}.json")


def download_all_gameweeks(session: requests.Session, num_gameweeks: int) -> None:
    """Download data for all gameweeks."""
    print(f"Downloading data for {num_gameweeks} gameweeks...")
    for gw in range(1, num_gameweeks + 1):
        download_gameweek_data(session, gw)
        time.sleep(0.2)


def download_league_data(session: requests.Session, league_id: int) -> None:
    """Download data for a specific league (classic or h2h)."""
    print(f"Downloading league {league_id} data...")
    data = fetch_endpoint(session, f"/leagues-classic/{league_id}/standings/")
    if data:
        save_json(data, DATA_DIR / "leagues" / f"classic_{league_id}.json")


def download_team_data(session: requests.Session, team_id: int) -> None:
    """Download data for a specific team."""
    print(f"Downloading team {team_id} data...")
    data = fetch_endpoint(session, f"/entry/{team_id}/")
    if data:
        save_json(data, DATA_DIR / "teams" / f"{team_id}.json")

    history = fetch_endpoint(session, f"/entry/{team_id}/history/")
    if history:
        save_json(history, DATA_DIR / "teams" / f"{team_id}_history.json")

    picks = fetch_endpoint(session, f"/entry/{team_id}/event/{1}/picks/")
    if picks:
        save_json(picks, DATA_DIR / "teams" / f"{team_id}_picks.json")


def main() -> None:
    """Main function to download all available FPL data."""
    print("=" * 60)
    print("FPL Data Downloader")
    print("=" * 60)

    session = create_session()

    bootstrap_data = download_bootstrap_static(session)
    if not bootstrap_data:
        print("Failed to download bootstrap-static data. Exiting.")
        return

    download_fixtures(session)

    if "elements" in bootstrap_data:
        player_ids = [player["id"] for player in bootstrap_data["elements"]]
        download_player_summaries(session, player_ids)

    if "events" in bootstrap_data:
        events = bootstrap_data["events"]
        if events:
            max_gameweek = max(e.get("id", 0) for e in events)
            if max_gameweek > 0:
                download_all_gameweeks(session, max_gameweek)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data saved to: {DATA_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

