"""
Script to calculate correlation-based weights for FPL midfielder metrics.

Analyzes all midfielders' historical data to determine which metrics correlate
most strongly with total points, then calculates optimal weights for the composite score.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DATA_DIR = Path("data")


def load_bootstrap_data() -> Dict:
    """Load bootstrap-static data."""
    with open(DATA_DIR / "bootstrap-static.json", "r") as f:
        return json.load(f)


def load_fixtures() -> List[Dict]:
    """Load fixtures data."""
    with open(DATA_DIR / "fixtures.json", "r") as f:
        return json.load(f)


def get_current_gameweek(bootstrap_data: Dict) -> int:
    """Get current gameweek number."""
    events = bootstrap_data.get("events", [])
    for event in events:
        if event.get("is_current"):
            return event.get("id", 1)
    return 1


def get_next_gameweek(bootstrap_data: Dict) -> int:
    """Get next gameweek number."""
    events = bootstrap_data.get("events", [])
    for event in events:
        if event.get("is_next"):
            return event.get("id", 1)
    current_gw = get_current_gameweek(bootstrap_data)
    return current_gw + 1


def calculate_team_points_from_fixtures(fixtures: List[Dict]) -> Dict[int, int]:
    """Calculate team points from finished fixtures."""
    team_points = {}
    for fixture in fixtures:
        if fixture.get("finished", False):
            team_h = fixture.get("team_h")
            team_a = fixture.get("team_a")
            score_h = fixture.get("team_h_score", 0)
            score_a = fixture.get("team_a_score", 0)

            if team_h not in team_points:
                team_points[team_h] = 0
            if team_a not in team_points:
                team_points[team_a] = 0

            if score_h > score_a:
                team_points[team_h] += 3
            elif score_a > score_h:
                team_points[team_a] += 3
            else:
                team_points[team_h] += 1
                team_points[team_a] += 1

    return team_points


def get_upcoming_fixtures(
    fixtures: List[Dict], team_id: int, start_gw: int, num_games: int = 5
) -> List[Dict]:
    """Get upcoming fixtures for a team starting from a specific gameweek."""
    upcoming = []
    for fixture in fixtures:
        if fixture.get("event") and fixture.get("event") >= start_gw:
            if fixture.get("team_h") == team_id or fixture.get("team_a") == team_id:
                upcoming.append(fixture)
                if len(upcoming) >= num_games:
                    break
    return sorted(upcoming, key=lambda x: x.get("event", 999))[:num_games]


def calculate_team_strength_score(
    team: Dict,
    team_points: Optional[Dict[int, int]] = None,
    team_id: Optional[int] = None,
) -> float:
    """Calculate team strength based on league position, points, and form."""
    position = team.get("position", 10)
    base_strength = 6.0 - (position - 1) * 0.25

    if team_points is not None and team_id is not None:
        points = team_points.get(team_id, 0)
    else:
        points = team.get("points", 0)

    points_adjustment = min(points / 100.0, 1.0) * 0.5

    form_str = team.get("form", "")
    form_score = 0.0
    if form_str:
        form_points = {"W": 3, "D": 1, "L": 0}
        recent_form = form_str[-5:] if len(form_str) >= 5 else form_str
        form_score = (
            sum(form_points.get(c, 0) for c in recent_form) / (len(recent_form) * 3)
            if recent_form
            else 0.0
        ) * 0.3

    return min(max(base_strength - points_adjustment - form_score, 1.0), 5.0)


def calculate_fixture_difficulty_score(
    fixtures: List[Dict],
    team_id: int,
    bootstrap_data: Dict,
    team_points: Optional[Dict[int, int]] = None,
) -> float:
    """Calculate average fixture difficulty using dynamic team strength."""
    if not fixtures:
        return 5.0

    teams_data = {t["id"]: t for t in bootstrap_data.get("teams", [])}
    difficulties = []

    for fixture in fixtures:
        if fixture.get("team_h") == team_id:
            opponent_id = fixture.get("team_a")
            is_home = True
        else:
            opponent_id = fixture.get("team_h")
            is_home = False

        opponent_team = teams_data.get(opponent_id, {})
        opponent_strength = calculate_team_strength_score(
            opponent_team, team_points, opponent_id
        )
        home_advantage = 0.3 if is_home else -0.3
        difficulty = opponent_strength - home_advantage
        difficulty = min(max(difficulty, 1.0), 5.0)
        difficulties.append(difficulty)

    return sum(difficulties) / len(difficulties) if difficulties else 5.0


def load_player_history(player_id: int) -> Optional[Dict]:
    """Load player history data."""
    player_file = DATA_DIR / "players" / f"{player_id}.json"
    if not player_file.exists():
        return None

    with open(player_file, "r") as f:
        return json.load(f)


def safe_float(value, default=0.0):
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def calculate_recent_form(
    player_history: Optional[Dict], current_gw: int, num_games: int = 5
) -> Dict:
    """Calculate recent form metrics from history."""
    if not player_history or not player_history.get("history"):
        return {
            "recent_xgi": 0.0,
            "recent_xg": 0.0,
            "recent_xa": 0.0,
            "recent_points": 0.0,
            "recent_minutes": 0,
            "games_played": 0,
            "avg_minutes_per_game": 0.0,
            "games_60plus_minutes": 0,
            "start_percentage": 0.0,
        }

    history = player_history["history"]
    recent_games = [h for h in history if h.get("round", 999) <= current_gw - 1]
    recent_games = sorted(recent_games, key=lambda x: x.get("round", 0), reverse=True)[
        :num_games
    ]

    if not recent_games:
        return {
            "recent_xgi": 0.0,
            "recent_xg": 0.0,
            "recent_xa": 0.0,
            "recent_points": 0.0,
            "recent_minutes": 0,
            "games_played": 0,
            "avg_minutes_per_game": 0.0,
            "games_60plus_minutes": 0,
            "start_percentage": 0.0,
        }

    total_xgi = sum(
        safe_float(h.get("expected_goal_involvements"), 0.0) for h in recent_games
    )
    total_xg = sum(safe_float(h.get("expected_goals"), 0.0) for h in recent_games)
    total_xa = sum(safe_float(h.get("expected_assists"), 0.0) for h in recent_games)
    total_points = sum(safe_float(h.get("total_points"), 0) for h in recent_games)
    total_minutes = sum(safe_float(h.get("minutes"), 0) for h in recent_games)
    games_played = len([h for h in recent_games if safe_float(h.get("minutes"), 0) > 0])

    games_60plus = len(
        [h for h in recent_games if safe_float(h.get("minutes"), 0) >= 60]
    )
    total_starts = sum(safe_float(h.get("starts"), 0) for h in recent_games)
    start_percentage = (total_starts / len(recent_games) * 100) if recent_games else 0.0
    avg_minutes = total_minutes / len(recent_games) if recent_games else 0.0

    return {
        "recent_xgi": total_xgi,
        "recent_xg": total_xg,
        "recent_xa": total_xa,
        "recent_points": total_points,
        "recent_minutes": int(total_minutes),
        "games_played": games_played,
        "avg_minutes_per_game": avg_minutes,
        "games_60plus_minutes": games_60plus,
        "start_percentage": start_percentage,
    }


def calculate_value_score(player: Dict, xgi_per_90: float) -> float:
    """Calculate value score (xGI per 90 / cost per million)."""
    cost = player.get("now_cost", 100) / 10.0
    if cost == 0:
        return 0.0
    return xgi_per_90 / cost


def calculate_metric_weights() -> Dict[str, float]:
    """
    Calculate correlation-based weights for each metric by analyzing all midfielders.

    Returns a dictionary mapping metric names to their weights based on correlation
    with total points scored.
    """
    print("Loading data...")
    bootstrap_data = load_bootstrap_data()
    fixtures = load_fixtures()
    current_gw = get_current_gameweek(bootstrap_data)
    next_gw = get_next_gameweek(bootstrap_data)

    players = bootstrap_data.get("elements", [])
    midfielders = [p for p in players if p.get("element_type") == 3]
    team_points = calculate_team_points_from_fixtures(fixtures)

    metric_data = {
        "xgi_per_90": [],
        "fdr_score": [],
        "form": [],
        "value_score": [],
        "rotation_score": [],
        "recent_xgi_per_game": [],
        "ict_index": [],
        "bonus_potential": [],
        "conversion_efficiency": [],
        "points": [],
    }

    print(f"Analyzing {len(midfielders)} midfielders...")
    processed = 0

    for player in midfielders:
        player_id = player.get("id")
        minutes = player.get("minutes", 0)
        if minutes < 180:
            continue

        player_history = load_player_history(player_id)
        recent_form = calculate_recent_form(player_history, current_gw)
        if recent_form["games_played"] < 3:
            continue

        team_id = player.get("team")
        upcoming_fixtures = get_upcoming_fixtures(
            fixtures, team_id, next_gw, num_games=5
        )
        avg_fdr = calculate_fixture_difficulty_score(
            upcoming_fixtures, team_id, bootstrap_data, team_points
        )

        xgi_per_90 = player.get("expected_goal_involvements_per_90", 0.0)
        form = float(player.get("form", 0.0))
        value_score = calculate_value_score(player, xgi_per_90)
        total_points = player.get("total_points", 0)

        ict_index = float(player.get("ict_index", 0.0))
        bonus_points = player.get("bonus", 0)
        bonus_potential = (
            (bonus_points / max(total_points, 1) * 100) if total_points > 0 else 0.0
        )

        goals_scored = player.get("goals_scored", 0)
        assists = player.get("assists", 0)
        gi = goals_scored + assists
        xgi = player.get("expected_goal_involvements", 0.0)
        conversion_efficiency = gi / xgi if xgi > 0 else 1.0

        games_60plus = recent_form["games_60plus_minutes"]
        games_played_recent = recent_form["games_played"]
        start_pct = recent_form["start_percentage"]
        pct_60plus = (
            (games_60plus / games_played_recent * 100) if games_played_recent > 0 else 0
        )
        rotation_risk_score = 1.0 - min(pct_60plus / 100.0, start_pct / 100.0)
        rotation_score = 1.0 - rotation_risk_score

        recent_xgi_per_game = recent_form["recent_xgi"] / max(
            recent_form["games_played"], 1
        )

        metric_data["xgi_per_90"].append(xgi_per_90)
        metric_data["fdr_score"].append(6.0 - avg_fdr)
        metric_data["form"].append(form)
        metric_data["value_score"].append(value_score)
        metric_data["rotation_score"].append(rotation_score)
        metric_data["recent_xgi_per_game"].append(recent_xgi_per_game)
        metric_data["ict_index"].append(ict_index)
        metric_data["bonus_potential"].append(bonus_potential)
        metric_data["conversion_efficiency"].append(conversion_efficiency)
        metric_data["points"].append(float(total_points))

        processed += 1

    print(f"Processed {processed} midfielders with sufficient data")

    if len(metric_data["points"]) < 10:
        print("Warning: Insufficient data, using default weights")
        return {
            "xgi_per_90": 3.0,
            "fdr_score": 0.5,
            "form": 0.3,
            "value_score": 2.0,
            "rotation_score": 0.5,
            "recent_xgi_per_game": 0.5,
            "ict_index": 0.1,
            "bonus_potential": 0.02,
            "conversion_efficiency": 0.4,
        }

    points_array = np.array(metric_data["points"])
    correlations = {}
    weights = {}

    print("\nMetric Correlations with Total Points:")
    print("-" * 50)

    for metric_name in [
        "xgi_per_90",
        "fdr_score",
        "form",
        "value_score",
        "rotation_score",
        "recent_xgi_per_game",
        "ict_index",
        "bonus_potential",
        "conversion_efficiency",
    ]:
        metric_array = np.array(metric_data[metric_name])
        if len(metric_array) > 0 and np.std(metric_array) > 0:
            correlation = np.corrcoef(metric_array, points_array)[0, 1]
            if not np.isnan(correlation):
                correlations[metric_name] = correlation
                weights[metric_name] = abs(correlation)
                print(f"{metric_name:25s}: {correlation:7.4f}")
            else:
                correlations[metric_name] = 0.0
                weights[metric_name] = 0.0
                print(f"{metric_name:25s}: {'N/A':>7s}")
        else:
            correlations[metric_name] = 0.0
            weights[metric_name] = 0.0
            print(f"{metric_name:25s}: {'N/A':>7s}")

    total_weight = sum(weights.values())
    if total_weight > 0:
        scale_factor = 10.0 / total_weight
        weights = {k: v * scale_factor for k, v in weights.items()}
    else:
        print("\nWarning: All correlations are zero, using default weights")
        weights = {
            "xgi_per_90": 3.0,
            "fdr_score": 0.5,
            "form": 0.3,
            "value_score": 2.0,
            "rotation_score": 0.5,
            "recent_xgi_per_game": 0.5,
            "ict_index": 0.1,
            "bonus_potential": 0.02,
            "conversion_efficiency": 0.4,
        }

    print("\nCalculated Weights:")
    print("-" * 50)
    for metric_name, weight in sorted(
        weights.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{metric_name:25s}: {weight:7.4f}")

    return weights


def main():
    """Main function."""
    weights = calculate_metric_weights()

    print("\n" + "=" * 50)
    print("Copy these weights to find_midfielders.py:")
    print("=" * 50)
    print()
    print("METRIC_WEIGHTS = {")
    for metric_name, weight in weights.items():
        print(f'    "{metric_name}": {weight:.4f},')
    print("}")
    print()


if __name__ == "__main__":
    main()
