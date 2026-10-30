"""Availability features from explicitly dated player snapshots."""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

FEATURE_NAMES = (
    "context_known", "context_age_8days", "available", "doubtful", "unavailable",
    "chance_known", "chance_next_round", "first_penalty_taker", "first_corner_taker",
)
HISTORICAL_FILE = Path(__file__).resolve().parent / "model_data/defender-context-2025-26.json"


def context_features(player, captured_at, cutoff):
    if captured_at is None:
        return [0] * len(FEATURE_NAMES)
    age = (cutoff - captured_at) / 86400
    if player is None or not 0 <= age <= 8:
        return [0] * len(FEATURE_NAMES)
    news = player.get("news_added")
    if news and datetime.fromisoformat(news.replace("Z", "+00:00")).timestamp() >= cutoff:
        return [0] * len(FEATURE_NAMES)
    chance = player.get("chance_of_playing_next_round")
    return [1, age / 8, int(player.get("status") == "a"), int(player.get("status") == "d"),
            int(player.get("status") in ("i", "s", "u")), int(chance is not None),
            float(chance or 0) / 100, int(player.get("penalties_order") == 1),
            int(player.get("corners_and_indirect_freekicks_order") == 1)]


@lru_cache(maxsize=1)
def historical_context():
    from historical_data import SOURCE_REVISION

    value = json.loads(HISTORICAL_FILE.read_text())
    if value["source_revision"] != SOURCE_REVISION or value["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Historical defender context has an incompatible source or schema.")
    return value["contexts"]


def add_training_context(samples, data_dir, season):
    snapshots = {}
    if season != "2025-26":
        for path in (Path(data_dir) / "forecast-ledger/snapshots" / season).glob("gw*.json"):
            value = json.loads(path.read_text())
            snapshots[value["origin"]] = value
    result = []
    for row in samples:
        if row["position"] != "DEF":
            result.append(row)
            continue
        if season == "2025-26":
            extra = historical_context().get(str(row["gw"]), {}).get(str(row["id"]), [0] * 9)
        else:
            snapshot = snapshots.get(row["gw"])
            extra = (context_features(snapshot["defender_context"].get(str(row["id"])),
                                      snapshot["bootstrap_captured_at"], row["cutoff"])
                     if snapshot else [0] * 9)
        if len(extra) != len(FEATURE_NAMES):
            raise ValueError("Invalid defender availability vector.")
        result.append({**row, "x": row["x"] + extra})
    return result
