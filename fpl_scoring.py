"""Shared scoring rules for the current FPL season.

Source: https://fantasy.premierleague.com/help/faqs (Defensive Contribution).
"""

from typing import Dict, List


def calculate_defcon_stats(history: List[Dict], element_type: int) -> Dict:
    """Award two points per qualifying appearance, capped at two per match."""
    threshold = {2: 10, 3: 12, 4: 12}.get(element_type)
    appearances = [match for match in history if (match.get("minutes") or 0) > 0]
    qualifying_games = sum(
        1
        for match in appearances
        if threshold is not None and float(match.get("defensive_contribution") or 0) >= threshold
    )
    return {
        "points": qualifying_games * 2,
        "threshold": threshold,
        "threshold_pct": qualifying_games / len(appearances) * 100 if appearances else 0.0,
    }
