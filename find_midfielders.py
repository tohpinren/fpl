"""
Script to find potential midfielders with good underlying stats for FPL transfers.

Analyzes expected goals involvement (xGI), fixture difficulty, form, and other key metrics.
Uses machine learning model to predict points over a configurable lookahead (default next 5 GWs).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpl_scoring import calculate_defcon_stats
from forecasting import predict_from_history, train_position_model


DATA_DIR = Path("data")

DEFAULT_LOOKAHEAD_GAMEWEEKS = 5

METRIC_WEIGHTS = {
    "xgi_per_90": 0.6303,
    "fdr_score": 0.3385,
    "form": 1.4768,
    "value_score": 0.3684,
    "rotation_score": 1.4657,
    "recent_xgi_per_game": 1.1756,
    "ict_index": 1.8310,
    "bonus_potential": 1.5263,
    "conversion_efficiency": 0.5365,
    "defcon_points_pct": 0.3115,
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
        opponent_strength = calculate_team_strength_score(opponent_team, team_points, opponent_id)

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
    recent_games = sorted(recent_games, key=lambda x: x.get("round", 0), reverse=True)[:num_games]

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

    total_xgi = sum(safe_float(h.get("expected_goal_involvements"), 0.0) for h in recent_games)
    total_xg = sum(safe_float(h.get("expected_goals"), 0.0) for h in recent_games)
    total_xa = sum(safe_float(h.get("expected_assists"), 0.0) for h in recent_games)
    total_points = sum(safe_float(h.get("total_points"), 0) for h in recent_games)
    total_minutes = sum(safe_float(h.get("minutes"), 0) for h in recent_games)
    games_played = len([h for h in recent_games if safe_float(h.get("minutes"), 0) > 0])

    games_60plus = len([h for h in recent_games if safe_float(h.get("minutes"), 0) >= 60])
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


def train_prediction_model(
    bootstrap_data: Dict,
    fixtures: List[Dict],
    current_gw: int,
    *,
    lookahead_gws: int,
    verbose: bool = False,
) -> Optional[Any]:
    """Fit the shared point-in-time model selected on historical development data."""
    return train_position_model(
        bootstrap_data, fixtures, DATA_DIR, "MID", horizon=lookahead_gws, verbose=verbose
    )


def predict_player_points(
    model: Any,
    player: Dict,
    player_history: Optional[Dict],
    prediction_start_gw: int,
    fixtures: List[Dict],
    bootstrap_data: Dict,
    team_points: Dict[int, int],
    recent_form: Dict,
    *,
    lookahead_gws: int,
) -> Optional[float]:
    """Predict a complete horizon with the same feature builder used in backtests."""
    return predict_from_history(
        model, player, player_history, prediction_start_gw, fixtures, bootstrap_data, lookahead_gws
    )


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
    *,
    lookahead_gws: int = DEFAULT_LOOKAHEAD_GAMEWEEKS,
    min_minutes: int = 180,
    min_games: int = 2,
    max_price: Optional[float] = None,
) -> List[Dict]:
    """Analyze all midfielders and return ranked list."""
    effective_next_gw = next_gw if next_gw is not None else get_next_gameweek(bootstrap_data)

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
        recent_form = calculate_recent_form(player_history, effective_next_gw)

        if recent_form["games_played"] < min_games:
            continue

        chance_playing = player.get("chance_of_playing_next_round", 100)
        status = player.get("status", "")
        news = player.get("news", "")

        avg_minutes_recent = recent_form["avg_minutes_per_game"]
        games_60plus = recent_form["games_60plus_minutes"]
        start_pct = recent_form["start_percentage"]

        if avg_minutes_recent < 50:
            continue

        is_injured = status == "i" or (chance_playing is not None and chance_playing < 75)

        team_id = player.get("team")
        upcoming_fixtures = get_upcoming_fixtures(
            fixtures,
            team_id,
            effective_next_gw,
            num_games=lookahead_gws,
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
        defcon = player.get("defensive_contribution", 0)
        defcon_per_90 = float(player.get("defensive_contribution_per_90", 0.0) or 0.0)
        clean_sheets = player.get("clean_sheets", 0)

        games_played_season = player.get("games_played", 0) or player.get("starts", 0) + player.get(
            "sub_appearances", 0
        )
        defcon_per_game = defcon / max(games_played_season, 1) if games_played_season > 0 else 0.0

        defcon_stats = calculate_defcon_stats(
            (player_history or {}).get("history", []), player.get("element_type")
        )
        defcon_points_pct = (
            defcon_stats["points"] / max(total_points, 1) * 100 if total_points > 0 else 0.0
        )
        defcon_threshold_pct = defcon_stats["threshold_pct"]

        rotation_risk_score = 0.0
        if recent_form["games_played"] >= 3:
            games_played_recent = recent_form["games_played"]
            pct_60plus = (
                (games_60plus / games_played_recent * 100) if games_played_recent > 0 else 0
            )
            rotation_risk_score = 1.0 - min(pct_60plus / 100.0, start_pct / 100.0)

        value_score = calculate_value_score(player, xgi_per_90)

        bonus_potential = (bonus_points / max(total_points, 1) * 100) if total_points > 0 else 0.0

        predicted_points = None
        if prediction_model is not None:
            predicted_points = predict_player_points(
                prediction_model,
                player,
                player_history,
                effective_next_gw,
                fixtures,
                bootstrap_data,
                team_points,
                recent_form,
                lookahead_gws=lookahead_gws,
            )

        fdr_score = 6.0 - avg_fdr
        rotation_score = 1.0 - rotation_risk_score
        recent_xgi_per_game = recent_form["recent_xgi"] / max(recent_form["games_played"], 1)

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
            + defcon_points_pct * METRIC_WEIGHTS["defcon_points_pct"]
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
            "defcon": defcon,
            "defcon_per_90": defcon_per_90,
            "defcon_per_game": defcon_per_game,
            "defcon_points_pct": defcon_points_pct,
            "defcon_threshold_pct": defcon_threshold_pct,
            "clean_sheets": clean_sheets,
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
    opponent_strength = calculate_team_strength_score(opponent_team, team_points, opponent_id)
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
    results: List[Dict],
    top_n: int = 30,
    team_points: Optional[Dict[int, int]] = None,
    *,
    lookahead_gws: int = DEFAULT_LOOKAHEAD_GAMEWEEKS,
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
                status_info += f" | Chance of playing: {player.get('chance_playing_next')}%"
            if player.get("news"):
                status_info += f" | News: {player.get('news', '')[:50]}"
            print(f"   ⚠️  {status_info}")

        print(f"   Composite Score: {player['composite_score']:.2f}")
        print(f"   Expected Goal Involvement per 90: {player['xgi_per_90']:.2f}")
        print(f"   - xG per 90: {player['xg_per_90']:.2f} | xA per 90: {player['xa_per_90']:.2f}")
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
        print(f"   Bonus Points: {player.get('bonus_points', 0)} | BPS: {player.get('bps', 0)}")
        print(f"   Ownership: {player['selected_by']:.1f}%")
        print(f"   Value Score (xGI/90 per £m): {player['value_score']:.3f}")
        print(f"   Average FDR (next {lookahead_gws}): {player['avg_fdr']:.1f} (lower is easier)")
        if player.get("defcon", 0) > 0:
            print(
                f"   Defcon: {player.get('defcon', 0)} total | {player.get('defcon_per_90', 0):.2f} per 90 | {player.get('defcon_per_game', 0):.2f} per game | {player.get('defcon_points_pct', 0):.1f}% of points | {player.get('defcon_threshold_pct', 0):.1f}% games >=12 defcon"
            )
            print(f"   Clean Sheets: {player.get('clean_sheets', 0)}")
        if player.get("predicted_points") is not None:
            print(
                f"   Predicted Points (next {lookahead_gws} GWs): {player['predicted_points']:.1f}"
            )

        print("   Upcoming Fixtures:")
        for fixture in player["upcoming_fixtures"][:lookahead_gws]:
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
    parser.add_argument(
        "--lookahead-gws",
        type=int,
        default=DEFAULT_LOOKAHEAD_GAMEWEEKS,
        metavar="N",
        help=(
            "How many upcoming gameweeks from the next GW to use for fixtures, "
            "FDR, ML targets/features, and predicted points "
            "(default: %(default)s; use e.g. 4 late in the season)"
        ),
    )
    args = parser.parse_args()

    if args.lookahead_gws < 1:
        parser.error("--lookahead-gws must be at least 1")

    print("Loading data...")
    bootstrap_data = load_bootstrap_data()
    fixtures = load_fixtures()

    current_gw = get_current_gameweek(bootstrap_data)
    next_gw = get_next_gameweek(bootstrap_data)
    print(f"Current Gameweek: {current_gw}")
    print(f"Next Gameweek (for transfers): {next_gw}")
    if args.max_price is not None:
        print(f"Maximum price filter: £{args.max_price:.1f}m")
    print(f"Lookahead gameweeks: {args.lookahead_gws}")
    print()

    print("Training prediction model...")
    prediction_model = train_prediction_model(
        bootstrap_data,
        fixtures,
        current_gw,
        lookahead_gws=args.lookahead_gws,
        verbose=True,
    )

    print("Analyzing midfielders...")
    results = analyze_midfielders(
        bootstrap_data,
        fixtures,
        current_gw,
        prediction_model=prediction_model,
        next_gw=next_gw,
        lookahead_gws=args.lookahead_gws,
        max_price=args.max_price,
    )

    team_points = calculate_team_points_from_fixtures(fixtures)

    print(f"Found {len(results)} midfielders meeting criteria")
    print()

    print_recommendations(
        results,
        top_n=30,
        team_points=team_points,
        lookahead_gws=args.lookahead_gws,
    )


if __name__ == "__main__":
    main()
