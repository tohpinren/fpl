"""Shared point-in-time FPL forecasts and purged chronological evaluation.

Features use only completed matches before the prediction cutoff. No current
bootstrap totals, ownership, FPL forecasts or final fixture difficulty enter X.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
POSITIONS = {2: "DEF", 3: "MID", 4: "FWD"}
FEATURE_NAMES = (
    "season_ppg",
    "recent_ppg_3",
    "recent_ppg_5",
    "recent_ppg_10",
    "recent_minutes",
    "recent_start_rate",
    "recent_60_rate",
    "appearance_rate",
    "season_xg90",
    "season_xa90",
    "recent_xg90",
    "recent_xa90",
    "recent_goals",
    "recent_assists",
    "recent_bonus",
    "recent_bps",
    "recent_ict",
    "recent_defcon",
    "recent_clean_sheets",
    "recent_goals_conceded",
    "season_minutes",
    "past_matches",
    "price",
    "fixture_count",
    "home_share",
    "opponent_ppg",
    "opponent_goals_against",
    "opponent_clean_sheet_rate",
    "team_ppg",
    "team_goals_for",
    "team_goals_against",
    "team_clean_sheet_rate",
    "naive_points",
    "minutes_baseline",
)
CANDIDATES = ("recent_baseline", "minutes_baseline", "ridge", "forest", "boosting", "ridge_blend")


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def number(row, key):
    return float(row.get(key) or 0)


def team_table(fixtures, start_gw, cutoff):
    table = {}
    for fixture in fixtures:
        if not fixture.get("finished") or (fixture.get("event") or 99) >= start_gw:
            continue
        if (
            not fixture.get("kickoff_time")
            or timestamp(fixture["kickoff_time"]) + 3 * 3600 >= cutoff
        ):
            continue
        if fixture.get("team_h_score") is None or fixture.get("team_a_score") is None:
            continue
        for home in (True, False):
            team = fixture["team_h" if home else "team_a"]
            goals = fixture["team_h_score" if home else "team_a_score"]
            conceded = fixture["team_a_score" if home else "team_h_score"]
            stats = table.setdefault(team, {"played": 0, "points": 0, "gf": 0, "ga": 0, "cs": 0})
            stats["played"] += 1
            stats["points"] += 3 if goals > conceded else 1 if goals == conceded else 0
            stats["gf"] += goals
            stats["ga"] += conceded
            stats["cs"] += int(conceded == 0)
    # Three pseudo-matches shrink early-season strengths towards neutral priors.
    return {
        team: {
            "ppg": (s["points"] + 4) / (s["played"] + 3),
            "gf": (s["gf"] + 4) / (s["played"] + 3),
            "ga": (s["ga"] + 4) / (s["played"] + 3),
            "cs": (s["cs"] + 1) / (s["played"] + 3),
        }
        for team, s in table.items()
    }


def prediction_cutoff(bootstrap, fixtures, start_gw):
    event = next((event for event in bootstrap.get("events", []) if event["id"] == start_gw), None)
    if event and event.get("deadline_time"):
        return timestamp(event["deadline_time"])
    dates = [
        timestamp(f["kickoff_time"])
        for f in fixtures
        if f.get("event") == start_gw and f.get("kickoff_time")
    ]
    return min(dates) - 90 * 60 if dates else float("inf")


def make_features(player, history, fixtures, start_gw, horizon, cutoff, table=None, live=False):
    past = sorted(
        [
            row
            for row in history
            if row.get("round", 99) < start_gw
            and row.get("kickoff_time")
            and timestamp(row["kickoff_time"]) + 3 * 3600 < cutoff
        ],
        key=lambda row: (row["kickoff_time"], row["fixture"]),
    )
    if len(past) < 3:
        return None
    recent = past[-5:]
    if not any(number(row, "minutes") > 0 for row in recent):
        return None
    fixture_by_id = {fixture["id"]: fixture for fixture in fixtures}
    last_fixture = fixture_by_id[past[-1]["fixture"]]
    team_id = (
        player["team"] if live else last_fixture["team_h" if past[-1]["was_home"] else "team_a"]
    )
    upcoming = [
        f
        for f in fixtures
        if start_gw <= (f.get("event") or 0) < start_gw + horizon
        and team_id in (f["team_h"], f["team_a"])
    ]
    neutral = {"ppg": 4 / 3, "gf": 4 / 3, "ga": 4 / 3, "cs": 1 / 3}
    table = team_table(fixtures, start_gw, cutoff) if table is None else table
    own = table.get(team_id, neutral)
    opponents = [
        table.get(f["team_a"] if f["team_h"] == team_id else f["team_h"], neutral) for f in upcoming
    ]
    mean = lambda rows, key: sum(number(row, key) for row in rows) / max(len(rows), 1)
    minutes = sum(number(row, "minutes") for row in past)
    recent_minutes = mean(recent, "minutes")
    # Per-match denominators include zero-minute absences. Rates per 90 shrink
    # towards the player's season rate rather than exploding after one cameo.
    xg90 = sum(number(row, "expected_goals") for row in past) * 90 / max(minutes, 90)
    xa90 = sum(number(row, "expected_assists") for row in past) * 90 / max(minutes, 90)
    recent_total_minutes = sum(number(row, "minutes") for row in recent)
    recent_xg90 = (
        (sum(number(row, "expected_goals") for row in recent) + xg90 * 2)
        * 90
        / (recent_total_minutes + 180)
    )
    recent_xa90 = (
        (sum(number(row, "expected_assists") for row in recent) + xa90 * 2)
        * 90
        / (recent_total_minutes + 180)
    )
    appearance_rate = mean(
        [{**row, "appeared": int(number(row, "minutes") > 0)} for row in recent], "appeared"
    )
    naive = mean(recent, "total_points") * len(upcoming)
    points_per90 = sum(number(row, "total_points") for row in past) * 90 / max(minutes, 90)
    minutes_baseline = points_per90 * recent_minutes / 90 * len(upcoming)
    features = [
        mean(past, "total_points"),
        mean(past[-3:], "total_points"),
        mean(recent, "total_points"),
        mean(past[-10:], "total_points"),
        recent_minutes,
        mean(recent, "starts"),
        sum(number(r, "minutes") >= 60 for r in recent) / len(recent),
        appearance_rate,
        xg90,
        xa90,
        recent_xg90,
        recent_xa90,
        mean(recent, "goals_scored"),
        mean(recent, "assists"),
        mean(recent, "bonus"),
        mean(recent, "bps"),
        mean(recent, "ict_index"),
        mean(recent, "defensive_contribution"),
        mean(recent, "clean_sheets"),
        mean(recent, "goals_conceded"),
        minutes,
        len(past),
        number(past[-1], "value") / 10,
        len(upcoming),
        sum(f["team_h"] == team_id for f in upcoming) / max(len(upcoming), 1),
        *(
            sum(opponent[key] for opponent in opponents) / max(len(opponents), 1)
            for key in ("ppg", "ga", "cs")
        ),
        *(own[key] for key in ("ppg", "gf", "ga", "cs")),
        naive,
        minutes_baseline,
    ]
    return {
        "x": features,
        "baseline": naive,
        "minutes_baseline": minutes_baseline,
        "eligible": minutes >= 180
        and recent_minutes >= 50
        and sum(number(r, "minutes") > 0 for r in recent) >= 2,
        "price": number(past[-1], "value") / 10,
        "team": team_id,
        "past": past,
        "fixture_count": len(upcoming),
    }


def make_samples(bootstrap, fixtures, histories, horizon=5, max_start=None):
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one gameweek.")
    completed = {event["id"] for event in bootstrap["events"] if event.get("finished")}
    samples = []
    for gw in sorted(completed):
        if max_start is not None and gw > max_start:
            continue
        target_gws = set(range(gw, gw + horizon))
        if not target_gws <= completed:
            continue
        target_fixtures = [f for f in fixtures if f.get("event") in target_gws]
        if not target_fixtures or not all(f.get("finished") for f in target_fixtures):
            continue
        cutoff = prediction_cutoff(bootstrap, fixtures, gw)
        label_end = max(timestamp(f["kickoff_time"]) for f in target_fixtures) + 3 * 3600
        table = team_table(fixtures, gw, cutoff)
        for player in bootstrap["elements"]:
            if player["element_type"] not in POSITIONS:
                continue
            history = histories.get(player["id"], [])
            features = make_features(player, history, fixtures, gw, horizon, cutoff, table)
            if features is None:
                continue
            # Missing appearances within a completed horizon score zero, including
            # injuries and departures. Do not discard low-scoring future players.
            target = sum(
                number(row, "total_points") for row in history if row["round"] in target_gws
            )
            samples.append(
                {
                    "id": player["id"],
                    "position": POSITIONS[player["element_type"]],
                    "gw": gw,
                    "cutoff": cutoff,
                    "label_end": label_end,
                    "target": target,
                    **{key: value for key, value in features.items() if key != "past"},
                }
            )
    return samples


def training_before(samples, origin, cutoff, horizon):
    return [row for row in samples if row["gw"] + horizon <= origin and row["label_end"] < cutoff]


def estimator(name):
    if name in ("ridge", "ridge_blend"):
        return make_pipeline(StandardScaler(), Ridge(alpha=100))
    if name == "forest":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.8,
            n_jobs=2,
            random_state=42,
        )
    if name == "boosting":
        return HistGradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=15,
            l2_regularization=10,
            min_samples_leaf=30,
            random_state=42,
        )
    return None


@dataclass
class PointModel:
    name: str
    model: object = None

    def predict(self, rows):
        x = np.asarray([row["x"] if isinstance(row, dict) else row for row in rows], dtype=float)
        baseline = x[:, FEATURE_NAMES.index("naive_points")]
        if self.name == "recent_baseline":
            prediction = baseline
        elif self.name == "minutes_baseline":
            prediction = x[:, FEATURE_NAMES.index("minutes_baseline")]
        else:
            prediction = self.model.predict(x)
            if self.name == "ridge_blend":
                prediction = 0.5 * prediction + 0.5 * baseline
        # Keep blank horizons at zero and constrain expected totals to nonnegative values.
        return np.where(
            x[:, FEATURE_NAMES.index("fixture_count")] == 0, 0, np.maximum(0, prediction)
        )


def fit_model(name, samples):
    model = estimator(name)
    if model is not None:
        model.fit(
            np.asarray([s["x"] for s in samples], dtype=float),
            np.asarray([s["target"] for s in samples]),
        )
    return PointModel(name, model)


def load_live_histories(data_dir, bootstrap):
    histories = {}
    for player in bootstrap["elements"]:
        path = Path(data_dir) / "players" / f"{player['id']}.json"
        if path.exists():
            histories[player["id"]] = json.loads(path.read_text()).get("history", [])
    return histories


@lru_cache(maxsize=4)
def historical_samples(horizon):
    from historical_data import load_history

    bootstrap, fixtures, histories, _ = load_history()
    return make_samples(bootstrap, fixtures, histories, horizon)


def train_position_model(bootstrap, fixtures, data_dir, position, horizon=5, verbose=False):
    """Fit a validation-selected configuration; never select using live/test targets."""
    from download_fpl_data import get_season

    config = json.loads((ROOT / "model_config.json").read_text())
    histories = load_live_histories(data_dir, bootstrap)
    next_event = next((event for event in bootstrap["events"] if event.get("is_next")), None)
    start = (
        next_event["id"]
        if next_event
        else max((event["id"] for event in bootstrap["events"] if event.get("finished")), default=0)
        + 1
    )
    cutoff = min(
        prediction_cutoff(bootstrap, fixtures, start), datetime.now(timezone.utc).timestamp()
    )
    samples = training_before(
        make_samples(bootstrap, fixtures, histories, horizon), start, cutoff, horizon
    )
    if get_season(bootstrap) != "2025-26":
        samples = historical_samples(horizon) + samples
    samples = [sample for sample in samples if sample["position"] == position]
    if not samples:
        return fit_model("recent_baseline", [])
    model = fit_model(config["models"][position], samples)
    if verbose:
        print(
            f"Using {position} {model.name}: {len(samples)} completed, past-only training samples; {horizon}-GW targets."
        )
    return model


def predict_from_history(model, player, history, start_gw, fixtures, bootstrap, horizon):
    cutoff = min(
        prediction_cutoff(bootstrap, fixtures, start_gw), datetime.now(timezone.utc).timestamp()
    )
    features = make_features(
        player, (history or {}).get("history", []), fixtures, start_gw, horizon, cutoff, live=True
    )
    return None if features is None else float(model.predict([features])[0])


def live_forecasts(bootstrap, fixtures, data_dir, horizon=5):
    """Train the preselected models on completed historical/current labels."""
    config = json.loads((ROOT / "model_config.json").read_text())
    historic_samples = historical_samples(horizon)
    histories = load_live_histories(data_dir, bootstrap)
    next_event = next((event for event in bootstrap["events"] if event.get("is_next")), None)
    if next_event is None:
        return {}, {"horizon": horizon, "models": {}, "training_season": "2025/26"}
    start = next_event["id"]
    cutoff = min(timestamp(next_event["deadline_time"]), datetime.now(timezone.utc).timestamp())
    current_samples = training_before(
        make_samples(bootstrap, fixtures, histories, horizon), start, cutoff, horizon
    )
    table = team_table(fixtures, start, cutoff)
    forecasts = {}
    for element_type, position in POSITIONS.items():
        train = [s for s in historic_samples + current_samples if s["position"] == position]
        model = fit_model(config["models"][position], train)
        for player in bootstrap["elements"]:
            if player["element_type"] != element_type:
                continue
            features = make_features(
                player,
                histories.get(player["id"], []),
                fixtures,
                start,
                horizon,
                cutoff,
                table,
                live=True,
            )
            if features is not None:
                forecasts[player["id"]] = {
                    "points": float(model.predict([features])[0]),
                    "model": model.name,
                }
    return forecasts, {
        "horizon": horizon,
        "models": config["models"],
        "training_season": "2025/26",
        "review": "reports/model-review-2025-26.md",
    }
