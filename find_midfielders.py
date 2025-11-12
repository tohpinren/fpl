"""
Script to find potential midfielders with good underlying stats for FPL transfers.

Analyzes expected goals involvement (xGI), fixture difficulty, form, and other key metrics.
Uses machine learning model to predict points in next 5 games.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None

DATA_DIR = Path("data")

METRIC_WEIGHTS = {
    "xgi_per_90": 3.0,
    "fdr_score": 0.5,
    "form": 0.3,
    "value_score": 2.0,
    "rotation_score": 0.5,
    "recent_xgi_per_game": 0.5,
    "ict_index": 0.1,
    "bonus_potential": 0.02,
    "conversion_efficiency": 0.4,
    "predicted_points": 0.15,
}


def load_bootstrap_data() -> Dict:
    """Load bootstrap-static data."""
    with open(DATA_DIR / "bootstrap-static.json", "r") as f:
        return json.load(f)


def load_fixtures() -> List[Dict]:
    """Load fixtures data."""
    with open(DATA_DIR / "fixtures.json", "r") as f:
        return json.load(f)


def get_current_gameweek(bootstrap_data: Dict) -> int:
    """Get the current gameweek number."""
    events = bootstrap_data.get("events", [])
    for event in events:
        if event.get("is_current"):
            return event.get("id", 1)
    return 1


def get_next_gameweek(bootstrap_data: Dict) -> int:
    """Get the next gameweek number for transfer planning."""
    events = bootstrap_data.get("events", [])
    for event in events:
        if event.get("is_next"):
            return event.get("id", 1)
    current_gw = get_current_gameweek(bootstrap_data)
    return current_gw + 1


def calculate_team_points_from_fixtures(fixtures: List[Dict]) -> Dict[int, int]:
    """Calculate team points from finished fixtures.

    Returns a dictionary mapping team_id to total points.
    """
    team_points = {}

    for fixture in fixtures:
        if not fixture.get("finished"):
            continue

        team_h = fixture.get("team_h")
        team_a = fixture.get("team_a")
        score_h = fixture.get("team_h_score")
        score_a = fixture.get("team_a_score")

        if team_h is None or team_a is None or score_h is None or score_a is None:
            continue

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
        event = fixture.get("event")
        if event is not None and event >= start_gw:
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
    """Calculate team strength score based on league table standings.

    Returns a score from 1-5 where:
    - 1 = Very weak (bottom teams)
    - 5 = Very strong (top teams)
    """
    position = team.get("position", 20)

    if team_points is not None and team_id is not None:
        points = team_points.get(team_id, 0)
    else:
        points = team.get("points", 0)

    played = team.get("played", 1)
    if played == 0:
        played = 1

    points_per_game = points / max(played, 1)

    form_str = team.get("form", "")
    form_score = 0.0
    if form_str:
        form_points = {"W": 3, "D": 1, "L": 0}
        recent_form = form_str[-5:] if len(form_str) >= 5 else form_str
        form_total = sum(form_points.get(c, 0) for c in recent_form)
        form_score = form_total / (len(recent_form) * 3) if recent_form else 0.0

    position_score = 1.0 + (21 - position) / 5.0
    points_score = 1.0 + (points_per_game / 3.0) * 2.0

    strength = position_score * 0.4 + points_score * 0.4 + form_score * 2.0 * 0.2
    return min(max(strength, 1.0), 5.0)


def calculate_fixture_difficulty_score(
    fixtures: List[Dict],
    team_id: int,
    bootstrap_data: Dict,
    team_points: Optional[Dict[int, int]] = None,
) -> float:
    """Calculate average fixture difficulty for upcoming games using league table standings.

    Lower score = easier fixture (opponent is weaker)
    Higher score = harder fixture (opponent is stronger)
    """
    if not fixtures:
        return 5.0

    teams_data = {t["id"]: t for t in bootstrap_data.get("teams", [])}

    difficulties = []
    for fixture in fixtures:
        opponent_id = None
        is_home = False

        if fixture.get("team_h") == team_id:
            opponent_id = fixture.get("team_a")
            is_home = True
        else:
            opponent_id = fixture.get("team_h")
            is_home = False

        if opponent_id is None:
            difficulties.append(5.0)
            continue

        opponent_team = teams_data.get(opponent_id, {})
        opponent_strength = calculate_team_strength_score(
            opponent_team, team_points, opponent_id
        )

        home_advantage = 0.3 if is_home else -0.3
        difficulty = opponent_strength - home_advantage

        difficulties.append(min(max(difficulty, 1.0), 5.0))

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


def get_player_features_for_prediction(
    player: Dict,
    player_history: Optional[Dict],
    current_gw: int,
    fixtures: List[Dict],
    bootstrap_data: Dict,
    team_points: Dict[int, int],
    recent_form: Dict,
) -> Optional[Dict]:
    """Extract comprehensive features for prediction model from player data."""
    if not player_history or not player_history.get("history"):
        return None

    history = player_history["history"]
    past_games = [h for h in history if h.get("round", 999) < current_gw]

    if len(past_games) < 3:
        return None

    past_games = sorted(past_games, key=lambda x: x.get("round", 0), reverse=True)
    recent_5 = past_games[:5]
    recent_3 = past_games[:3]
    recent_10 = past_games[:10]

    def calc_recent_stats(games_list):
        games_played = len(
            [h for h in games_list if safe_float(h.get("minutes"), 0) > 0]
        )
        if games_played == 0:
            return {
                "total_xgi": 0.0,
                "total_xg": 0.0,
                "total_xa": 0.0,
                "total_points": 0.0,
                "total_goals": 0,
                "total_assists": 0,
                "total_bonus": 0,
                "total_bps": 0,
                "total_minutes": 0,
                "total_starts": 0,
                "games_played": 0,
                "home_games": 0,
                "home_points": 0.0,
                "away_games": 0,
                "away_points": 0.0,
            }

        total_xgi = sum(
            safe_float(h.get("expected_goal_involvements"), 0.0) for h in games_list
        )
        total_xg = sum(safe_float(h.get("expected_goals"), 0.0) for h in games_list)
        total_xa = sum(safe_float(h.get("expected_assists"), 0.0) for h in games_list)
        total_points = sum(safe_float(h.get("total_points"), 0) for h in games_list)
        total_goals = sum(safe_float(h.get("goals_scored"), 0) for h in games_list)
        total_assists = sum(safe_float(h.get("assists"), 0) for h in games_list)
        total_bonus = sum(safe_float(h.get("bonus"), 0) for h in games_list)
        total_bps = sum(safe_float(h.get("bps"), 0) for h in games_list)
        total_minutes = sum(safe_float(h.get("minutes"), 0) for h in games_list)
        total_starts = sum(safe_float(h.get("starts"), 0) for h in games_list)

        home_games = [
            h
            for h in games_list
            if h.get("was_home") and safe_float(h.get("minutes"), 0) > 0
        ]
        away_games = [
            h
            for h in games_list
            if not h.get("was_home") and safe_float(h.get("minutes"), 0) > 0
        ]
        home_points = sum(safe_float(h.get("total_points"), 0) for h in home_games)
        away_points = sum(safe_float(h.get("total_points"), 0) for h in away_games)

        return {
            "total_xgi": total_xgi,
            "total_xg": total_xg,
            "total_xa": total_xa,
            "total_points": total_points,
            "total_goals": total_goals,
            "total_assists": total_assists,
            "total_bonus": total_bonus,
            "total_bps": total_bps,
            "total_minutes": total_minutes,
            "total_starts": total_starts,
            "games_played": games_played,
            "home_games": len(home_games),
            "home_points": home_points,
            "away_games": len(away_games),
            "away_points": away_points,
        }

    stats_5 = calc_recent_stats(recent_5)
    stats_3 = calc_recent_stats(recent_3)
    stats_10 = calc_recent_stats(recent_10)

    if stats_5["games_played"] == 0:
        return None

    games_60plus = len([h for h in recent_5 if safe_float(h.get("minutes"), 0) >= 60])
    pct_60plus = (
        (games_60plus / stats_5["games_played"] * 100)
        if stats_5["games_played"] > 0
        else 0.0
    )
    start_pct = (stats_5["total_starts"] / len(recent_5) * 100) if recent_5 else 0.0
    avg_minutes = stats_5["total_minutes"] / len(recent_5) if recent_5 else 0.0

    team_id = player.get("team")
    teams_data = {t["id"]: t for t in bootstrap_data.get("teams", [])}
    player_team = teams_data.get(team_id, {})

    upcoming_fixtures = []
    for fixture in fixtures:
        event = fixture.get("event")
        if event is not None and current_gw <= event < current_gw + 5:
            if fixture.get("team_h") == team_id or fixture.get("team_a") == team_id:
                upcoming_fixtures.append(fixture)

    if len(upcoming_fixtures) == 0:
        return None

    difficulties = []
    home_fixtures = 0
    for fixture in upcoming_fixtures:
        if fixture.get("team_h") == team_id:
            opponent_id = fixture.get("team_a")
            is_home = True
            home_fixtures += 1
        else:
            opponent_id = fixture.get("team_h")
            is_home = False

        opponent_team = teams_data.get(opponent_id, {})
        opponent_pos = opponent_team.get("position", 20)
        opponent_points = team_points.get(opponent_id) if opponent_id is not None else 0

        position_score = 1.0 + (21 - opponent_pos) / 5.0
        points_score = (
            1.0 + (opponent_points / max(opponent_team.get("played", 1), 1) / 3.0) * 2.0
        )
        opponent_strength = position_score * 0.5 + points_score * 0.5

        home_advantage = 0.3 if is_home else -0.3
        difficulty = opponent_strength - home_advantage
        difficulties.append(min(max(difficulty, 1.0), 5.0))

    avg_fdr = sum(difficulties) / len(difficulties) if difficulties else 5.0
    home_fixture_pct = (
        (home_fixtures / len(upcoming_fixtures) * 100) if upcoming_fixtures else 0.0
    )

    xgi_per_90 = player.get("expected_goal_involvements_per_90", 0.0)
    xg_per_90 = player.get("expected_goals_per_90", 0.0)
    xa_per_90 = player.get("expected_assists_per_90", 0.0)
    form = float(player.get("form", 0.0))
    ict_index = float(player.get("ict_index", 0.0))
    influence = float(player.get("influence", 0.0))
    creativity = float(player.get("creativity", 0.0))
    threat = float(player.get("threat", 0.0))
    cost = player.get("now_cost", 0) / 10.0
    points_per_game = float(player.get("points_per_game", 0.0))
    total_points = player.get("total_points", 0)
    bonus = player.get("bonus", 0)
    bps = player.get("bps", 0)
    goals_scored = player.get("goals_scored", 0)
    assists = player.get("assists", 0)
    selected_by = float(player.get("selected_by_percent", 0.0))
    transfers_in = player.get("transfers_in", 0)
    transfers_out = player.get("transfers_out", 0)
    transfers_in_event = player.get("transfers_in_event", 0)
    transfers_out_event = player.get("transfers_out_event", 0)
    ep_next = float(player.get("ep_next", 0.0))
    ep_this = float(player.get("ep_this", 0.0))
    value_form = float(player.get("value_form", 0.0))
    value_season = float(player.get("value_season", 0.0))
    minutes = player.get("minutes", 0)
    starts = player.get("starts", 0)
    yellow_cards = player.get("yellow_cards", 0)
    red_cards = player.get("red_cards", 0)

    team_strength = calculate_team_strength_score(player_team, team_points, team_id)
    team_form_str = player_team.get("form", "")
    team_form_score = 0.0
    if team_form_str:
        form_points = {"W": 3, "D": 1, "L": 0}
        recent_form = team_form_str[-5:] if len(team_form_str) >= 5 else team_form_str
        team_form_score = (
            sum(form_points.get(c, 0) for c in recent_form) / (len(recent_form) * 3)
            if recent_form
            else 0.0
        )

    rotation_risk = 1.0 - min(pct_60plus / 100.0, start_pct / 100.0)
    bonus_potential = (bonus / max(total_points, 1) * 100) if total_points > 0 else 0.0
    transfer_trend = (
        (transfers_in_event - transfers_out_event) / max(selected_by, 1)
        if selected_by > 0
        else 0.0
    )
    transfer_ratio = (
        transfers_in / max(transfers_out, 1)
        if transfers_out > 0
        else (transfers_in if transfers_in > 0 else 0.0)
    )

    home_ppg = (
        stats_5["home_points"] / max(stats_5["home_games"], 1)
        if stats_5["home_games"] > 0
        else 0.0
    )
    away_ppg = (
        stats_5["away_points"] / max(stats_5["away_games"], 1)
        if stats_5["away_games"] > 0
        else 0.0
    )

    features = {
        "xgi_per_90": xgi_per_90,
        "xg_per_90": xg_per_90,
        "xa_per_90": xa_per_90,
        "form": form,
        "points_per_game": points_per_game,
        "ict_index": ict_index,
        "influence": influence,
        "creativity": creativity,
        "threat": threat,
        "cost": cost,
        "avg_fdr": avg_fdr,
        "rotation_risk": rotation_risk,
        "avg_minutes_recent": avg_minutes,
        "start_pct": start_pct,
        "pct_60plus": pct_60plus,
        "recent_xgi_per_game_5": stats_5["total_xgi"] / max(stats_5["games_played"], 1),
        "recent_xg_per_game_5": stats_5["total_xg"] / max(stats_5["games_played"], 1),
        "recent_xa_per_game_5": stats_5["total_xa"] / max(stats_5["games_played"], 1),
        "recent_points_per_game_5": stats_5["total_points"]
        / max(stats_5["games_played"], 1),
        "recent_goals_per_game_5": stats_5["total_goals"]
        / max(stats_5["games_played"], 1),
        "recent_assists_per_game_5": stats_5["total_assists"]
        / max(stats_5["games_played"], 1),
        "recent_bonus_per_game_5": stats_5["total_bonus"]
        / max(stats_5["games_played"], 1),
        "recent_bps_per_game_5": stats_5["total_bps"] / max(stats_5["games_played"], 1),
        "recent_points_per_game_3": (
            stats_3["total_points"] / max(stats_3["games_played"], 1)
            if stats_3["games_played"] > 0
            else 0.0
        ),
        "recent_xgi_per_game_3": (
            stats_3["total_xgi"] / max(stats_3["games_played"], 1)
            if stats_3["games_played"] > 0
            else 0.0
        ),
        "recent_points_per_game_10": (
            stats_10["total_points"] / max(stats_10["games_played"], 1)
            if stats_10["games_played"] > 0
            else 0.0
        ),
        "total_minutes": stats_5["total_minutes"],
        "total_starts": stats_5["total_starts"],
        "games_played": stats_5["games_played"],
        "home_ppg": home_ppg,
        "away_ppg": away_ppg,
        "home_fixture_pct": home_fixture_pct,
        "team_strength": team_strength,
        "team_form": team_form_score,
        "bonus_potential": bonus_potential,
        "selected_by": selected_by,
        "transfer_trend": transfer_trend,
        "transfer_ratio": transfer_ratio,
        "ep_next": ep_next,
        "ep_this": ep_this,
        "value_form": value_form,
        "value_season": value_season,
        "total_goals": goals_scored,
        "total_assists": assists,
        "total_bonus": bonus,
        "total_bps": bps,
        "season_minutes": minutes,
        "season_starts": starts,
        "yellow_cards": yellow_cards,
        "red_cards": red_cards,
    }

    return features


def get_actual_points_next_5_games(
    player_history: Optional[Dict], current_gw: int
) -> Optional[int]:
    """Get actual points scored in next 5 games from current gameweek."""
    if not player_history or not player_history.get("history"):
        return None

    history = player_history["history"]
    future_games = [
        h
        for h in history
        if h.get("round", 999) >= current_gw and h.get("round", 999) < current_gw + 5
    ]

    if len(future_games) == 0:
        return None

    total_points = sum(safe_float(h.get("total_points"), 0) for h in future_games)
    return int(total_points)


def calculate_team_points_from_fixtures_for_gw(
    fixtures: List[Dict], up_to_gw: Optional[int] = None
) -> Dict[int, int]:
    """Calculate team points from finished fixtures up to a specific gameweek."""
    team_points = {}

    for fixture in fixtures:
        if not fixture.get("finished"):
            continue

        event = fixture.get("event")
        if up_to_gw is not None and event is not None and event > up_to_gw:
            continue

        team_h = fixture.get("team_h")
        team_a = fixture.get("team_a")
        score_h = fixture.get("team_h_score")
        score_a = fixture.get("team_a_score")

        if team_h is None or team_a is None or score_h is None or score_a is None:
            continue

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


def prepare_training_data(
    bootstrap_data: Dict,
    fixtures: List[Dict],
    min_gw: int = 3,
    max_gw: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
    """Prepare training data from historical gameweeks."""
    players = bootstrap_data.get("elements", [])
    midfielders = [p for p in players if p.get("element_type") == 3]

    if max_gw is None:
        events = bootstrap_data.get("events", [])
        finished_events = [e for e in events if e.get("finished")]
        if finished_events:
            max_gw = max(e.get("id", 0) for e in finished_events)
        else:
            max_gw = min_gw

    X = []
    y = []

    for gw in range(min_gw, max_gw + 1):
        team_points = calculate_team_points_from_fixtures_for_gw(
            fixtures, up_to_gw=gw - 1
        )

        for player in midfielders:
            player_id = player.get("id")
            player_history = load_player_history(player_id)

            recent_form = calculate_recent_form(player_history, gw - 1)
            features = get_player_features_for_prediction(
                player,
                player_history,
                gw,
                fixtures,
                bootstrap_data,
                team_points,
                recent_form,
            )

            if features is None:
                continue

            actual_points = get_actual_points_next_5_games(player_history, gw)

            if actual_points is None:
                continue

            feature_vector = [
                features["xgi_per_90"],
                features["xg_per_90"],
                features["xa_per_90"],
                features["form"],
                features["points_per_game"],
                features["ict_index"],
                features["influence"],
                features["creativity"],
                features["threat"],
                features["cost"],
                features["avg_fdr"],
                features["rotation_risk"],
                features["avg_minutes_recent"],
                features["start_pct"],
                features["pct_60plus"],
                features["recent_xgi_per_game_5"],
                features["recent_xg_per_game_5"],
                features["recent_xa_per_game_5"],
                features["recent_points_per_game_5"],
                features["recent_goals_per_game_5"],
                features["recent_assists_per_game_5"],
                features["recent_bonus_per_game_5"],
                features["recent_bps_per_game_5"],
                features["recent_points_per_game_3"],
                features["recent_xgi_per_game_3"],
                features["recent_points_per_game_10"],
                features["total_minutes"],
                features["total_starts"],
                features["games_played"],
                features["home_ppg"],
                features["away_ppg"],
                features["home_fixture_pct"],
                features["team_strength"],
                features["team_form"],
                features["bonus_potential"],
                features["selected_by"],
                features["transfer_trend"],
                features["transfer_ratio"],
                features["ep_next"],
                features["ep_this"],
                features["value_form"],
                features["value_season"],
                features["total_goals"],
                features["total_assists"],
                features["total_bonus"],
                features["total_bps"],
                features["season_minutes"],
                features["season_starts"],
                features["yellow_cards"],
                features["red_cards"],
            ]

            X.append(feature_vector)
            y.append(actual_points)

    return X, y


def train_and_evaluate_model(
    model, model_name: str, X_train, X_test, y_train, y_test, feature_names: List[str]
) -> Dict:
    """Train and evaluate a model, returning performance metrics."""
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_corr = np.corrcoef(y_train, y_pred_train)[0, 1] if len(y_train) > 1 else 0.0
    test_corr = np.corrcoef(y_test, y_pred_test)[0, 1] if len(y_test) > 1 else 0.0

    metrics = {
        "name": model_name,
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "train_r2": r2_score(y_train, y_pred_train),
        "train_corr": train_corr if not np.isnan(train_corr) else 0.0,
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "test_r2": r2_score(y_test, y_pred_test),
        "test_corr": test_corr if not np.isnan(test_corr) else 0.0,
        "model": model,
    }

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        metrics["feature_importance"] = feature_importance

    return metrics


def train_prediction_model(
    bootstrap_data: Dict, fixtures: List[Dict], current_gw: int, verbose: bool = False
) -> Optional[Any]:
    """Train prediction model and return the best model."""
    if verbose:
        print("Preparing training data for prediction model...")
    X, y = prepare_training_data(bootstrap_data, fixtures, min_gw=3, max_gw=None)

    if len(X) < 50:
        if verbose:
            print(
                "Not enough training data for prediction model. Skipping ML predictions."
            )
        return None

    if verbose:
        print(f"Training samples: {len(X)}")

    X_array = np.array(X)
    y_array = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=0.2, random_state=42
    )

    feature_names = [
        "xgi_per_90",
        "xg_per_90",
        "xa_per_90",
        "form",
        "points_per_game",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "cost",
        "avg_fdr",
        "rotation_risk",
        "avg_minutes_recent",
        "start_pct",
        "pct_60plus",
        "recent_xgi_per_game_5",
        "recent_xg_per_game_5",
        "recent_xa_per_game_5",
        "recent_points_per_game_5",
        "recent_goals_per_game_5",
        "recent_assists_per_game_5",
        "recent_bonus_per_game_5",
        "recent_bps_per_game_5",
        "recent_points_per_game_3",
        "recent_xgi_per_game_3",
        "recent_points_per_game_10",
        "total_minutes",
        "total_starts",
        "games_played",
        "home_ppg",
        "away_ppg",
        "home_fixture_pct",
        "team_strength",
        "team_form",
        "bonus_potential",
        "selected_by",
        "transfer_trend",
        "transfer_ratio",
        "ep_next",
        "ep_this",
        "value_form",
        "value_season",
        "total_goals",
        "total_assists",
        "total_bonus",
        "total_bps",
        "season_minutes",
        "season_starts",
        "yellow_cards",
        "red_cards",
    ]

    models_to_train = []

    models_to_train.append(
        (
            RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "Random Forest",
        )
    )

    models_to_train.append((Ridge(alpha=2.0, random_state=42), "Ridge Regression"))

    if XGBOOST_AVAILABLE:
        models_to_train.append(
            (
                XGBRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    random_state=42,
                    n_jobs=-1,
                ),
                "XGBoost",
            )
        )

    if LIGHTGBM_AVAILABLE:
        models_to_train.append(
            (
                LGBMRegressor(
                    n_estimators=150,
                    max_depth=7,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=1.5,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
                "LightGBM",
            )
        )

    results = []

    for model, name in models_to_train:
        metrics = train_and_evaluate_model(
            model, name, X_train, X_test, y_train, y_test, feature_names
        )
        results.append(metrics)

    if verbose:
        print("Model Performance:")
        for r in results:
            print(
                f"  {r['name']}: Test MAE={r['test_mae']:.2f}, R²={r['test_r2']:.3f}, "
                f"Correlation={r['test_corr']:.3f}"
            )

    best_model_metrics = min(results, key=lambda x: x["test_mae"])

    if verbose:
        print(f"Using best model: {best_model_metrics['name']}")
        print()

    return best_model_metrics["model"]


def predict_player_points(
    model: Any,
    player: Dict,
    player_history: Optional[Dict],
    prediction_start_gw: int,
    fixtures: List[Dict],
    bootstrap_data: Dict,
    team_points: Dict[int, int],
    recent_form: Dict,
) -> Optional[float]:
    """Predict points for next 5 games starting from prediction_start_gw.

    The model predicts total points for prediction_start_gw to prediction_start_gw + 4.
    """
    features = get_player_features_for_prediction(
        player,
        player_history,
        prediction_start_gw,
        fixtures,
        bootstrap_data,
        team_points,
        recent_form,
    )

    if features is None:
        return None

    feature_vector = [
        features["xgi_per_90"],
        features["xg_per_90"],
        features["xa_per_90"],
        features["form"],
        features["points_per_game"],
        features["ict_index"],
        features["influence"],
        features["creativity"],
        features["threat"],
        features["cost"],
        features["avg_fdr"],
        features["rotation_risk"],
        features["avg_minutes_recent"],
        features["start_pct"],
        features["pct_60plus"],
        features["recent_xgi_per_game_5"],
        features["recent_xg_per_game_5"],
        features["recent_xa_per_game_5"],
        features["recent_points_per_game_5"],
        features["recent_goals_per_game_5"],
        features["recent_assists_per_game_5"],
        features["recent_bonus_per_game_5"],
        features["recent_bps_per_game_5"],
        features["recent_points_per_game_3"],
        features["recent_xgi_per_game_3"],
        features["recent_points_per_game_10"],
        features["total_minutes"],
        features["total_starts"],
        features["games_played"],
        features["home_ppg"],
        features["away_ppg"],
        features["home_fixture_pct"],
        features["team_strength"],
        features["team_form"],
        features["bonus_potential"],
        features["selected_by"],
        features["transfer_trend"],
        features["transfer_ratio"],
        features["ep_next"],
        features["ep_this"],
        features["value_form"],
        features["value_season"],
        features["total_goals"],
        features["total_assists"],
        features["total_bonus"],
        features["total_bps"],
        features["season_minutes"],
        features["season_starts"],
        features["yellow_cards"],
        features["red_cards"],
    ]

    predicted_points = model.predict([feature_vector])[0]
    return float(predicted_points)


def calculate_value_score(player: Dict, xgi_per_90: float) -> float:
    """Calculate value score (xGI per 90 / cost per million)."""
    cost = player.get("now_cost", 100) / 10.0
    if cost == 0:
        return 0.0
    return xgi_per_90 / cost


def analyze_midfielders(
    bootstrap_data: Dict,
    fixtures: List[Dict],
    current_gw: int,
    prediction_model: Optional[Any] = None,
    next_gw: Optional[int] = None,
    min_minutes: int = 180,
    min_games: int = 3,
    max_price: Optional[float] = None,
) -> List[Dict]:
    """Analyze all midfielders and return ranked list."""
    if next_gw is None:
        next_gw = current_gw + 1

    players = bootstrap_data.get("elements", [])
    teams = {t["id"]: t["name"] for t in bootstrap_data.get("teams", [])}

    team_points = calculate_team_points_from_fixtures(fixtures)

    midfielders = [p for p in players if p.get("element_type") == 3]

    results = []

    for player in midfielders:
        player_id = player.get("id")
        minutes = player.get("minutes", 0)
        cost = player.get("now_cost", 0) / 10.0

        if minutes < min_minutes:
            continue

        if max_price is not None and cost > max_price:
            continue

        player_history = load_player_history(player_id)
        recent_form = calculate_recent_form(player_history, current_gw)

        if recent_form["games_played"] < min_games:
            continue

        chance_playing = player.get("chance_of_playing_next_round", 100)
        status = player.get("status", "")
        news = player.get("news", "")

        avg_minutes_recent = recent_form["avg_minutes_per_game"]
        games_60plus = recent_form["games_60plus_minutes"]
        start_pct = recent_form["start_percentage"]

        is_injured = status == "i" or (
            chance_playing is not None and chance_playing < 75
        )

        if not is_injured:
            if recent_form["games_played"] >= 3:
                games_played_recent = recent_form["games_played"]
                pct_60plus = (
                    (games_60plus / games_played_recent * 100)
                    if games_played_recent > 0
                    else 0
                )

                if avg_minutes_recent < 60 or pct_60plus < 60 or start_pct < 80:
                    continue

        team_id = player.get("team")
        next_gw = get_next_gameweek(bootstrap_data)
        upcoming_fixtures = get_upcoming_fixtures(
            fixtures, team_id, next_gw, num_games=5
        )
        avg_fdr = calculate_fixture_difficulty_score(
            upcoming_fixtures, team_id, bootstrap_data, team_points
        )

        xg = float(player.get("expected_goals", 0.0) or 0.0)
        xa = float(player.get("expected_assists", 0.0) or 0.0)
        xgi = float(player.get("expected_goal_involvements", 0.0) or 0.0)
        xg_per_90 = float(player.get("expected_goals_per_90", 0.0) or 0.0)
        xa_per_90 = float(player.get("expected_assists_per_90", 0.0) or 0.0)
        xgi_per_90 = float(player.get("expected_goal_involvements_per_90", 0.0) or 0.0)

        goals_scored = int(player.get("goals_scored", 0) or 0)
        assists = int(player.get("assists", 0) or 0)
        gi = goals_scored + assists

        conversion_efficiency = gi / xgi if xgi > 0 else 1.0

        cost = player.get("now_cost", 0) / 10.0
        form = float(player.get("form", 0.0))
        points_per_game = float(player.get("points_per_game", 0.0))
        total_points = player.get("total_points", 0)
        selected_by = float(player.get("selected_by_percent", 0.0))

        starts = player.get("starts", 0)

        ict_index = float(player.get("ict_index", 0.0))
        influence = float(player.get("influence", 0.0))
        creativity = float(player.get("creativity", 0.0))
        threat = float(player.get("threat", 0.0))

        bonus_points = player.get("bonus", 0)
        bps = player.get("bps", 0)

        rotation_risk_score = 0.0
        if recent_form["games_played"] >= 3:
            games_played_recent = recent_form["games_played"]
            pct_60plus = (
                (games_60plus / games_played_recent * 100)
                if games_played_recent > 0
                else 0
            )
            rotation_risk_score = 1.0 - min(pct_60plus / 100.0, start_pct / 100.0)

        value_score = calculate_value_score(player, xgi_per_90)

        bonus_potential = (
            (bonus_points / max(total_points, 1) * 100) if total_points > 0 else 0.0
        )

        predicted_points = None
        if prediction_model is not None:
            predicted_points = predict_player_points(
                prediction_model,
                player,
                player_history,
                next_gw,
                fixtures,
                bootstrap_data,
                team_points,
                recent_form,
            )

        fdr_score = 6.0 - avg_fdr
        rotation_score = 1.0 - rotation_risk_score
        recent_xgi_per_game = recent_form["recent_xgi"] / max(
            recent_form["games_played"], 1
        )

        composite_score = (
            xgi_per_90 * METRIC_WEIGHTS["xgi_per_90"]
            + fdr_score * METRIC_WEIGHTS["fdr_score"]
            + form * METRIC_WEIGHTS["form"]
            + value_score * METRIC_WEIGHTS["value_score"]
            + rotation_score * METRIC_WEIGHTS["rotation_score"]
            + recent_xgi_per_game * METRIC_WEIGHTS["recent_xgi_per_game"]
            + ict_index * METRIC_WEIGHTS["ict_index"]
            + bonus_potential * METRIC_WEIGHTS["bonus_potential"]
            + conversion_efficiency * METRIC_WEIGHTS["conversion_efficiency"]
        )

        if predicted_points is not None:
            composite_score += predicted_points * METRIC_WEIGHTS["predicted_points"]

        result = {
            "id": player_id,
            "name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "team": teams.get(team_id, "Unknown"),
            "team_id": team_id,
            "cost": cost,
            "xgi_per_90": xgi_per_90,
            "xg_per_90": xg_per_90,
            "xa_per_90": xa_per_90,
            "total_xgi": xgi,
            "total_xg": xg,
            "total_xa": xa,
            "goals": goals_scored,
            "assists": assists,
            "gi": gi,
            "conversion_efficiency": conversion_efficiency,
            "form": form,
            "points_per_game": points_per_game,
            "total_points": total_points,
            "minutes": minutes,
            "starts": starts,
            "rotation_risk": rotation_risk_score,
            "selected_by": selected_by,
            "ict_index": ict_index,
            "influence": influence,
            "creativity": creativity,
            "threat": threat,
            "avg_fdr": avg_fdr,
            "upcoming_fixtures": upcoming_fixtures,
            "value_score": value_score,
            "composite_score": composite_score,
            "recent_form": recent_form,
            "bonus_points": bonus_points,
            "bps": bps,
            "is_injured": is_injured,
            "chance_playing_next": chance_playing,
            "status": status,
            "news": news,
            "predicted_points": predicted_points,
        }

        results.append(result)

    return sorted(results, key=lambda x: x["composite_score"], reverse=True)


def format_fixture(
    fixture: Dict,
    team_id: int,
    teams: Dict,
    bootstrap_data: Dict,
    team_points: Optional[Dict[int, int]] = None,
) -> str:
    """Format fixture information with dynamic FDR based on league standings."""
    teams_data = {t["id"]: t for t in bootstrap_data.get("teams", [])}

    if fixture.get("team_h") == team_id:
        opponent_id = fixture.get("team_a")
        opponent = teams.get(opponent_id, "Unknown")
        home_away = "H"
        is_home = True
    else:
        opponent_id = fixture.get("team_h")
        opponent = teams.get(opponent_id, "Unknown")
        home_away = "A"
        is_home = False

    opponent_team = teams_data.get(opponent_id, {})
    opponent_strength = calculate_team_strength_score(
        opponent_team, team_points, opponent_id
    )
    home_advantage = 0.3 if is_home else -0.3
    difficulty = opponent_strength - home_advantage
    difficulty = min(max(difficulty, 1.0), 5.0)

    opponent_pos = opponent_team.get("position", "?")
    if team_points is not None and opponent_id is not None:
        opponent_points = team_points.get(opponent_id, 0)
    else:
        opponent_points = opponent_team.get("points", 0)

    gw = fixture.get("event", "?")

    return f"GW{gw} {home_away} vs {opponent} (Pos: {opponent_pos}, Pts: {opponent_points}, FDR: {difficulty:.1f})"


def print_recommendations(
    results: List[Dict], top_n: int = 20, team_points: Optional[Dict[int, int]] = None
):
    """Print formatted recommendations."""
    bootstrap_data = json.load(open(DATA_DIR / "bootstrap-static.json"))
    teams = {t["id"]: t["name"] for t in bootstrap_data.get("teams", [])}

    if team_points is None:
        fixtures = load_fixtures()
        team_points = calculate_team_points_from_fixtures(fixtures)

    print("=" * 100)
    print("TOP MIDFIELDER TRANSFER TARGETS - DATA-DRIVEN ANALYSIS")
    print("=" * 100)
    print()

    for i, player in enumerate(results[:top_n], 1):
        print(f"{i}. {player['name']} ({player['team']}) - £{player['cost']:.1f}m")

        chance_playing_next = player.get("chance_playing_next")
        if player.get("is_injured") or (
            chance_playing_next is not None and chance_playing_next < 100
        ):
            status_info = f"Status: {player.get('status', 'Unknown')}"
            if player.get("chance_playing_next") is not None:
                status_info += (
                    f" | Chance of playing: {player.get('chance_playing_next')}%"
                )
            if player.get("news"):
                status_info += f" | News: {player.get('news', '')[:50]}"
            print(f"   ⚠️  {status_info}")

        print(f"   Composite Score: {player['composite_score']:.2f}")
        print(f"   Expected Goal Involvement per 90: {player['xgi_per_90']:.2f}")
        print(
            f"   - xG per 90: {player['xg_per_90']:.2f} | xA per 90: {player['xa_per_90']:.2f}"
        )
        print(
            f"   Conversion Efficiency (GI/xGI): {player['conversion_efficiency']:.2f} (GI: {player['gi']}, xGI: {player['total_xgi']:.2f})"
        )
        print(
            f"   Form: {player['form']:.1f} | PPG: {player['points_per_game']:.1f} | Total Points: {player['total_points']}"
        )

        recent = player["recent_form"]
        if recent["games_played"] > 0:
            avg_mins = recent.get("avg_minutes_per_game", 0)
            pct_60plus = (
                (recent.get("games_60plus_minutes", 0) / recent["games_played"] * 100)
                if recent["games_played"] > 0
                else 0
            )
            start_pct = recent.get("start_percentage", 0)
            print(
                f"   Recent Minutes: Avg {avg_mins:.0f} min/game | {pct_60plus:.0f}% games 60+ min | {start_pct:.0f}% starts"
            )
            print(
                f"   Recent Form (last 5 games): {recent['games_played']} games, {recent['recent_points']} pts, {recent['recent_xgi']:.2f} xGI"
            )

        print(
            f"   Rotation Risk: {player['rotation_risk']:.1%} | Minutes: {player['minutes']} | Starts: {player['starts']}"
        )
        print(
            f"   ICT Index: {player['ict_index']:.1f} (Influence: {player['influence']:.1f}, Creativity: {player['creativity']:.1f}, Threat: {player['threat']:.1f})"
        )
        print(
            f"   Bonus Points: {player.get('bonus_points', 0)} | BPS: {player.get('bps', 0)}"
        )
        print(f"   Ownership: {player['selected_by']:.1f}%")
        print(f"   Value Score (xGI/90 per £m): {player['value_score']:.3f}")
        print(f"   Average FDR (next 5): {player['avg_fdr']:.1f} (lower is easier)")
        if player.get("predicted_points") is not None:
            print(
                f"   Predicted Points (next 5 games): {player['predicted_points']:.1f}"
            )

        print("   Upcoming Fixtures:")
        for fixture in player["upcoming_fixtures"][:5]:
            print(
                f"     - {format_fixture(fixture, player['team_id'], teams, bootstrap_data, team_points)}"
            )

        print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Find potential FPL midfielders with good underlying stats"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Maximum price in millions (e.g., 8.5 for £8.5m)",
    )
    args = parser.parse_args()

    print("Loading data...")
    bootstrap_data = load_bootstrap_data()
    fixtures = load_fixtures()

    current_gw = get_current_gameweek(bootstrap_data)
    next_gw = get_next_gameweek(bootstrap_data)
    print(f"Current Gameweek: {current_gw}")
    print(f"Next Gameweek (for transfers): {next_gw}")
    if args.max_price is not None:
        print(f"Maximum price filter: £{args.max_price:.1f}m")
    print()

    print("Training prediction model...")
    prediction_model = train_prediction_model(
        bootstrap_data, fixtures, current_gw, verbose=True
    )

    print("Analyzing midfielders...")
    results = analyze_midfielders(
        bootstrap_data,
        fixtures,
        current_gw,
        prediction_model=prediction_model,
        next_gw=next_gw,
        max_price=args.max_price,
    )

    team_points = calculate_team_points_from_fixtures(fixtures)

    print(f"Found {len(results)} midfielders meeting criteria")
    print()

    print_recommendations(results, top_n=20, team_points=team_points)


if __name__ == "__main__":
    main()
