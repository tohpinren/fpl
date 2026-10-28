"""
Script to find potential defenders with good underlying stats for FPL transfers.

FPL defender scoring: 4 pts clean sheet (60+ min), 2 pts for 10+ defensive contributions per match;
6 pts per goal, 3 per assist; 1 pt for 1-60 min, 2 pts for 60+; -1 per 2 goals conceded;
bonus up to 3 (BPS). Metrics and weights align to these: clean sheets, defcon, CS rate,
attacking xGI for goal/assist potential, rotation (60+), bonus. Also uses team defensive
strength: season-long clean sheets, goals conceded, and rates (CS rate, GA per game).
Uses machine learning model to predict points in next 5 games.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpl_scoring import calculate_defcon_stats
from forecasting import predict_from_history, train_position_model

from find_midfielders import (
    DEFAULT_LOOKAHEAD_GAMEWEEKS,
    calculate_fixture_difficulty_score,
    calculate_recent_form,
    calculate_team_points_from_fixtures,
    format_fixture,
    get_upcoming_fixtures,
    load_bootstrap_data,
    load_fixtures,
    load_player_history,
    safe_float,
)

DATA_DIR = Path("data")

DEFENDER_ELEMENT_TYPE = 2

# Weights aligned to FPL defender scoring: clean sheet 4 pts, defcon capped at 2 pts per match for 10+,
# goal 6 pts, assist 3 pts; appearance 1/2 pts; bonus up to 3.
# team_clean_sheet_rate / team_goals_conceded_per_game = season-long team defensive strength.
METRIC_WEIGHTS = {
    "defcon_per_90": 2.2,
    "clean_sheets_per_game": 2.8,
    "fdr_score": 0.34,
    "form": 1.2,
    "value_score": 0.5,
    "rotation_score": 1.6,
    "recent_xgi_per_game": 0.55,
    "ict_index": 1.0,
    "bonus_potential": 1.3,
    "defcon_points_pct": 0.45,
    "team_clean_sheet_rate": 1.4,
    "team_goals_conceded_per_game": -0.35,
    "predicted_points": 0.15,
}


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


def get_team_defensive_stats(
    fixtures: List[Dict], team_id: int, up_to_gw: Optional[int] = None
) -> Dict[str, float]:
    """
    Team defensive stats from finished fixtures: clean sheets, goals conceded,
    and rates over the season up to the given gameweek.

    Returns clean_sheets, goals_conceded, matches_played, clean_sheet_rate (0-1),
    goals_conceded_per_game. Rates are 0 if no matches played.
    """
    clean_sheets = 0
    goals_conceded = 0
    matches_played = 0

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

        if team_id == team_h:
            ga = score_a
            matches_played += 1
            if ga == 0:
                clean_sheets += 1
            goals_conceded += ga
        elif team_id == team_a:
            ga = score_h
            matches_played += 1
            if ga == 0:
                clean_sheets += 1
            goals_conceded += ga

    if matches_played == 0:
        return {
            "clean_sheets": 0,
            "goals_conceded": 0,
            "matches_played": 0,
            "clean_sheet_rate": 0.0,
            "goals_conceded_per_game": 0.0,
        }

    return {
        "clean_sheets": clean_sheets,
        "goals_conceded": goals_conceded,
        "matches_played": matches_played,
        "clean_sheet_rate": clean_sheets / matches_played,
        "goals_conceded_per_game": goals_conceded / matches_played,
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
        bootstrap_data, fixtures, DATA_DIR, "DEF", horizon=lookahead_gws, verbose=verbose
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


def calculate_value_score_defender(player: Dict, points_per_game: float) -> float:
    """Calculate value score for defenders (points per game per million)."""
    cost = player.get("now_cost", 100) / 10.0
    if cost == 0:
        return 0.0
    return points_per_game / cost


def recent_clean_sheets_per_game(
    player_history: Optional[Dict], current_gw: int, num_games: int = 5
) -> float:
    """Clean sheets per game over the last num_games."""
    if not player_history or not player_history.get("history"):
        return 0.0

    history = player_history["history"]
    recent = [h for h in history if h.get("round", 999) <= current_gw - 1]
    recent = sorted(recent, key=lambda x: x.get("round", 0), reverse=True)[:num_games]

    if not recent:
        return 0.0

    total_cs = sum(safe_float(h.get("clean_sheets"), 0) for h in recent)
    games_played = len([h for h in recent if safe_float(h.get("minutes"), 0) > 0])
    return total_cs / max(games_played, 1)


def analyze_defenders(
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
    """Analyze all defenders and return ranked list."""
    effective_next_gw = next_gw if next_gw is not None else get_next_gameweek(bootstrap_data)

    players = bootstrap_data.get("elements", [])
    teams = {t["id"]: t["name"] for t in bootstrap_data.get("teams", [])}

    team_points = calculate_team_points_from_fixtures(fixtures)

    defenders = [p for p in players if p.get("element_type") == DEFENDER_ELEMENT_TYPE]

    results = []

    for player in defenders:
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

        clean_sheets = int(player.get("clean_sheets", 0) or 0)
        defcon = player.get("defensive_contribution", 0)
        defcon_per_90 = float(player.get("defensive_contribution_per_90", 0.0) or 0.0)
        games_played_season = player.get("games_played", 0) or (
            player.get("starts", 0) + player.get("sub_appearances", 0)
        )
        clean_sheets_per_game = (
            clean_sheets / max(games_played_season, 1) if games_played_season > 0 else 0.0
        )
        defcon_per_game = defcon / max(games_played_season, 1) if games_played_season > 0 else 0.0

        total_points = player.get("total_points", 0)
        defcon_stats = calculate_defcon_stats(
            (player_history or {}).get("history", []), player.get("element_type")
        )
        defcon_points_pct = (
            defcon_stats["points"] / max(total_points, 1) * 100 if total_points > 0 else 0.0
        )
        defcon_threshold_pct = defcon_stats["threshold_pct"]

        xgi_per_90 = float(player.get("expected_goal_involvements_per_90", 0.0) or 0.0)
        form = float(player.get("form", 0.0))
        points_per_game = float(player.get("points_per_game", 0.0))
        ict_index = float(player.get("ict_index", 0.0))
        bonus_points = player.get("bonus", 0)
        bonus_potential = (bonus_points / max(total_points, 1) * 100) if total_points > 0 else 0.0

        rotation_risk_score = 0.0
        if recent_form["games_played"] >= 3:
            games_played_recent = recent_form["games_played"]
            pct_60plus = (
                (games_60plus / games_played_recent * 100) if games_played_recent > 0 else 0
            )
            rotation_risk_score = 1.0 - min(pct_60plus / 100.0, start_pct / 100.0)

        value_score = calculate_value_score_defender(player, points_per_game)

        recent_cs_per_game = recent_clean_sheets_per_game(
            player_history, effective_next_gw, num_games=5
        )

        team_defensive = get_team_defensive_stats(fixtures, team_id, up_to_gw=current_gw)
        team_clean_sheet_rate = team_defensive["clean_sheet_rate"]
        team_goals_conceded_per_game = team_defensive["goals_conceded_per_game"]

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
            defcon_per_90 * METRIC_WEIGHTS["defcon_per_90"]
            + clean_sheets_per_game * METRIC_WEIGHTS["clean_sheets_per_game"]
            + recent_cs_per_game * METRIC_WEIGHTS["clean_sheets_per_game"] * 0.6
            + fdr_score * METRIC_WEIGHTS["fdr_score"]
            + form * METRIC_WEIGHTS["form"]
            + value_score * METRIC_WEIGHTS["value_score"]
            + rotation_score * METRIC_WEIGHTS["rotation_score"]
            + recent_xgi_per_game * METRIC_WEIGHTS["recent_xgi_per_game"]
            + ict_index * METRIC_WEIGHTS["ict_index"]
            + bonus_potential * METRIC_WEIGHTS["bonus_potential"]
            + defcon_points_pct * METRIC_WEIGHTS["defcon_points_pct"]
            + team_clean_sheet_rate * METRIC_WEIGHTS["team_clean_sheet_rate"]
            + team_goals_conceded_per_game * METRIC_WEIGHTS["team_goals_conceded_per_game"]
        )

        if predicted_points is not None:
            composite_score += predicted_points * METRIC_WEIGHTS["predicted_points"]

        result = {
            "id": player_id,
            "name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "team": teams.get(team_id, "Unknown"),
            "team_id": team_id,
            "cost": cost,
            "clean_sheets": clean_sheets,
            "clean_sheets_per_game": clean_sheets_per_game,
            "recent_clean_sheets_per_game": recent_cs_per_game,
            "defcon": defcon,
            "defcon_per_90": defcon_per_90,
            "defcon_per_game": defcon_per_game,
            "defcon_points_pct": defcon_points_pct,
            "defcon_threshold_pct": defcon_threshold_pct,
            "form": form,
            "points_per_game": points_per_game,
            "total_points": total_points,
            "minutes": minutes,
            "starts": player.get("starts", 0),
            "rotation_risk": rotation_risk_score,
            "selected_by": float(player.get("selected_by_percent", 0.0)),
            "ict_index": ict_index,
            "avg_fdr": avg_fdr,
            "upcoming_fixtures": upcoming_fixtures,
            "value_score": value_score,
            "composite_score": composite_score,
            "recent_form": recent_form,
            "bonus_points": bonus_points,
            "bps": player.get("bps", 0),
            "is_injured": is_injured,
            "chance_playing_next": chance_playing,
            "status": status,
            "news": news,
            "predicted_points": predicted_points,
            "xgi_per_90": xgi_per_90,
            "goals": player.get("goals_scored", 0),
            "assists": player.get("assists", 0),
            "team_clean_sheets_season": team_defensive["clean_sheets"],
            "team_goals_conceded_season": team_defensive["goals_conceded"],
            "team_clean_sheet_rate": team_clean_sheet_rate,
            "team_goals_conceded_per_game": team_goals_conceded_per_game,
        }

        results.append(result)

    return sorted(results, key=lambda x: x["composite_score"], reverse=True)


def print_recommendations(
    results: List[Dict],
    top_n: int = 30,
    team_points: Optional[Dict[int, int]] = None,
    *,
    lookahead_gws: int = DEFAULT_LOOKAHEAD_GAMEWEEKS,
) -> None:
    """Print formatted defender recommendations."""
    bootstrap_data = json.load(open(DATA_DIR / "bootstrap-static.json"))
    teams = {t["id"]: t["name"] for t in bootstrap_data.get("teams", [])}

    if team_points is None:
        fixtures = load_fixtures()
        team_points = calculate_team_points_from_fixtures(fixtures)

    print("=" * 100)
    print("TOP DEFENDER TRANSFER TARGETS - DATA-DRIVEN ANALYSIS")
    print("=" * 100)
    print()

    for i, player in enumerate(results[:top_n], 1):
        print(f"{i}. {player['name']} ({player['team']}) - {player['cost']:.1f}m")

        chance_playing_next = player.get("chance_playing_next")
        if player.get("is_injured") or (
            chance_playing_next is not None and chance_playing_next < 100
        ):
            status_info = f"Status: {player.get('status', 'Unknown')}"
            if player.get("chance_playing_next") is not None:
                status_info += f" | Chance of playing: {player.get('chance_playing_next')}%"
            if player.get("news"):
                status_info += f" | News: {player.get('news', '')[:50]}"
            print(f"   {status_info}")

        print(f"   Composite Score: {player['composite_score']:.2f}")
        print(
            f"   Clean Sheets: {player['clean_sheets']} | Per game: {player['clean_sheets_per_game']:.2f} | Recent (last 5): {player['recent_clean_sheets_per_game']:.2f}/game"
        )
        print(
            f"   Team defensive (season): CS {player.get('team_clean_sheets_season', 0)} | GA {player.get('team_goals_conceded_season', 0)} | CS rate {player.get('team_clean_sheet_rate', 0):.2f} | GA/game {player.get('team_goals_conceded_per_game', 0):.2f}"
        )
        print(
            f"   Defcon: {player['defcon']} total | {player['defcon_per_90']:.2f} per 90 | {player['defcon_per_game']:.2f} per game | {player['defcon_points_pct']:.1f}% of points | {player.get('defcon_threshold_pct', 0):.1f}% games >=10 defcon"
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
                f"   Recent Form (last 5): {recent['games_played']} games, {recent['recent_points']} pts"
            )

        print(
            f"   Rotation Risk: {player['rotation_risk']:.1%} | Minutes: {player['minutes']} | Starts: {player['starts']}"
        )
        print(f"   ICT Index: {player['ict_index']:.1f}")
        print(f"   Bonus Points: {player.get('bonus_points', 0)} | BPS: {player.get('bps', 0)}")
        print(f"   Ownership: {player['selected_by']:.1f}%")
        print(f"   Value Score (PPG per million): {player['value_score']:.3f}")
        print(f"   Average FDR (next {lookahead_gws}): {player['avg_fdr']:.1f} (lower is easier)")
        if player.get("xgi_per_90", 0) > 0:
            print(
                f"   xGI per 90: {player['xgi_per_90']:.2f} | G: {player.get('goals', 0)} A: {player.get('assists', 0)}"
            )
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


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Find potential FPL defenders with good underlying stats"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Maximum price in millions (e.g., 6.0 for 6.0m)",
    )
    parser.add_argument(
        "--lookahead-gws",
        type=int,
        default=DEFAULT_LOOKAHEAD_GAMEWEEKS,
        metavar="N",
        help=(
            "How many upcoming gameweeks from the next GW to use for fixtures, "
            "FDR, ML targets/features, and predicted points "
            "(default: %(default)s)"
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
        print(f"Maximum price filter: {args.max_price:.1f}m")
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

    print("Analyzing defenders...")
    results = analyze_defenders(
        bootstrap_data,
        fixtures,
        current_gw,
        prediction_model=prediction_model,
        next_gw=next_gw,
        lookahead_gws=args.lookahead_gws,
        max_price=args.max_price,
    )

    team_points = calculate_team_points_from_fixtures(fixtures)

    print(f"Found {len(results)} defenders meeting criteria")
    print()

    print_recommendations(
        results,
        top_n=30,
        team_points=team_points,
        lookahead_gws=args.lookahead_gws,
    )


if __name__ == "__main__":
    main()
