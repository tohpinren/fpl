"""Serve the local FPL dashboard with the existing analysis scripts.

Run: python serve.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FRONTEND_DIR = ROOT / "frontend"
REFRESH_LOCK = Lock()


def build_dashboard():
    """Adapt local analysis results for the browser without training models per request."""
    with (DATA_DIR / "bootstrap-static.json").open(encoding="utf-8") as stream:
        bootstrap = json.load(stream)
    with (DATA_DIR / "fixtures.json").open(encoding="utf-8") as stream:
        fixtures = json.load(stream)

    import find_defenders
    import find_forwards
    import find_midfielders
    from download_fpl_data import get_season

    for module in (find_defenders, find_forwards, find_midfielders):
        module.DATA_DIR = DATA_DIR
    teams = {team["id"]: team for team in bootstrap["teams"]}
    players = {player["id"]: player for player in bootstrap["elements"]}
    next_event = next((event for event in bootstrap["events"] if event.get("is_next")), None)
    current_gw = find_midfielders.get_current_gameweek(bootstrap)
    next_gw = next_event["id"] if next_event else current_gw + 1
    output = []
    for position, analyze in (
        ("DEF", find_defenders.analyze_defenders),
        ("MID", find_midfielders.analyze_midfielders),
        ("FWD", find_forwards.analyze_forwards),
    ):
        # Recent form uses an exclusive cutoff; include the latest completed GW.
        results = analyze(bootstrap, fixtures, next_gw, next_gw=next_gw)
        for rank, result in enumerate(results, 1):
            player = players[result["id"]]
            team_id = player["team"]
            history = find_midfielders.load_player_history(player["id"]) or {}
            player_fixtures = []
            for fixture in sorted(fixtures, key=lambda item: (item.get("event") or 99, item["id"])):
                if fixture.get("finished") or (fixture.get("event") or 0) < next_gw:
                    continue
                if team_id not in (fixture["team_h"], fixture["team_a"]):
                    continue
                home = fixture["team_h"] == team_id
                opponent = teams[fixture["team_a"] if home else fixture["team_h"]]
                player_fixtures.append(
                    {
                        "gameweek": fixture["event"],
                        "opponent": opponent["short_name"],
                        "opponent_name": opponent["name"],
                        "home": home,
                        "difficulty": fixture["team_h_difficulty" if home else "team_a_difficulty"],
                    }
                )
                if len(player_fixtures) == 5:
                    break
            recent = sorted(
                [match for match in history.get("history", []) if match.get("round", 99) < next_gw],
                key=lambda match: (match["round"], match["fixture"]),
            )[-5:]
            output.append(
                {
                    "id": player["id"],
                    "name": player["web_name"],
                    "full_name": result["name"],
                    "team": teams[team_id]["short_name"],
                    "team_name": teams[team_id]["name"],
                    "position": position,
                    "rank": rank,
                    "score": round(result["composite_score"], 1),
                    "price": result["cost"],
                    "form": result["form"],
                    "points": result["total_points"],
                    "points_per_game": result["points_per_game"],
                    "xgi_per_90": result["xgi_per_90"],
                    "ownership": result["selected_by"],
                    "minutes": player["minutes"],
                    "goals": result["goals"],
                    "assists": result["assists"],
                    "clean_sheets": result["clean_sheets"],
                    "defcon_points_pct": result["defcon_points_pct"],
                    "status": player.get("status", "a"),
                    "news": player.get("news", ""),
                    "chance_playing": player.get("chance_of_playing_next_round"),
                    "fixtures": player_fixtures,
                    "recent": [
                        {"gameweek": match["round"], "points": match["total_points"]}
                        for match in recent
                    ],
                }
            )
    updated = datetime.fromtimestamp(
        (DATA_DIR / "bootstrap-static.json").stat().st_mtime, timezone.utc
    )
    return {
        "season": get_season(bootstrap).replace("-", "/"),
        "gameweek": next_gw if next_event else None,
        "deadline": next_event["deadline_time"] if next_event else None,
        "updated_at": updated.isoformat(),
        "players": output,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path != "/api/dashboard":
            return super().do_GET()
        if not REFRESH_LOCK.acquire(blocking=False):
            return self.send_json(
                {"error": "FPL data is refreshing. Please try again shortly."}, 409
            )
        try:
            self.send_json(build_dashboard())
        except FileNotFoundError:
            self.send_json(
                {"error": "Download FPL data first, or use Refresh data to get started."}, 503
            )
        except Exception as error:
            self.log_error("Failed to build dashboard: %s", error)
            self.send_json(
                {
                    "error": "Could not read the analysis data. Check the server terminal for details."
                },
                500,
            )
        finally:
            REFRESH_LOCK.release()

    def do_POST(self):
        if urlsplit(self.path).path != "/api/refresh":
            return self.send_json({"error": "Not found"}, 404)
        # Restrict writes to requests from this local page.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send_json({"error": "Refresh is only available from this dashboard."}, 403)
        if not REFRESH_LOCK.acquire(blocking=False):
            return self.send_json(
                {"error": "A data refresh is already running. Try again shortly."}, 409
            )
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "download_fpl_data.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if (
                result.returncode
                or "Error fetching" in result.stdout
                or "Failed to download" in result.stdout
            ):
                self.log_error("Data refresh failed: %s", result.stdout + result.stderr)
                return self.send_json(
                    {"error": "Some FPL data could not be downloaded. Please try again."}, 502
                )
            self.send_json(build_dashboard())
        except subprocess.TimeoutExpired:
            self.send_json(
                {"error": "The FPL API took too long to respond. Please try again."}, 504
            )
        except Exception as error:
            self.log_error("Data refresh failed: %s", error)
            self.send_json(
                {"error": "Could not refresh data. Check the server terminal for details."}, 500
            )
        finally:
            REFRESH_LOCK.release()


def main():
    parser = argparse.ArgumentParser(description="Open the local FPL dashboard")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    handler = partial(DashboardHandler, directory=str(FRONTEND_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"FPL dashboard: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
