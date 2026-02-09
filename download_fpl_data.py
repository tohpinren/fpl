"""
Script to download all available FPL (Fantasy Premier League) API data for the current season.

This script fetches data from the official FPL API endpoints and saves them locally.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://fantasy.premierleague.com/api"
DATA_DIR = Path("data")
MAX_WORKERS = 20
POOL_MAXSIZE = 32
RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)


def create_session() -> requests.Session:
    """Create a requests session with retry strategy and connection pool for parallel requests."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_maxsize=POOL_MAXSIZE)
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


def _fetch_and_save(
    session: requests.Session, endpoint: str, filepath: Path
) -> Tuple[bool, str]:
    """Fetch endpoint and save to filepath. Returns (success, endpoint) for use in executor."""
    data = fetch_endpoint(session, endpoint)
    if data:
        save_json(data, filepath)
        return True, endpoint
    return False, endpoint


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
    """Download element-summary for each player in parallel."""
    print(f"Downloading player summaries for {len(player_ids)} players...")
    players_dir = DATA_DIR / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    workers = min(MAX_WORKERS, len(player_ids))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_and_save,
                session,
                f"/element-summary/{player_id}/",
                players_dir / f"{player_id}.json",
            ): player_id
            for player_id in player_ids
        }
        done = 0
        for future in as_completed(futures):
            success, _ = future.result()
            if success:
                done += 1
            if done % 50 == 0 and done > 0:
                print(f"Progress: {done}/{len(player_ids)} players downloaded")
    print(f"Player summaries: {done}/{len(player_ids)} downloaded")


def download_gameweek_data(session: requests.Session, gameweek: int) -> None:
    """Download data for a specific gameweek."""
    data = fetch_endpoint(session, f"/event/{gameweek}/live/")
    if data:
        save_json(data, DATA_DIR / "gameweeks" / f"{gameweek}.json")


def download_all_gameweeks(session: requests.Session, num_gameweeks: int) -> None:
    """Download data for all gameweeks in parallel."""
    print(f"Downloading data for {num_gameweeks} gameweeks...")
    (DATA_DIR / "gameweeks").mkdir(parents=True, exist_ok=True)
    workers = min(MAX_WORKERS, num_gameweeks)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_gameweek_data, session, gw)
            for gw in range(1, num_gameweeks + 1)
        ]
        for f in as_completed(futures):
            f.result()
    print(f"Gameweeks: {num_gameweeks} downloaded")


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

    player_ids: List[int] = (
        [int(p["id"]) for p in bootstrap_data["elements"]]
        if "elements" in bootstrap_data
        else []
    )
    events = bootstrap_data.get("events") or []
    max_gameweek = max((e.get("id", 0) for e in events), default=0)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download_fixtures, session)]
        if player_ids:
            futures.append(executor.submit(download_player_summaries, session, player_ids))
        if max_gameweek > 0:
            futures.append(executor.submit(download_all_gameweeks, session, max_gameweek))
        for f in as_completed(futures):
            f.result()

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data saved to: {DATA_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

