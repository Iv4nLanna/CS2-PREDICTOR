import math
from datetime import datetime


def compute_head_to_head(
    matches: list[dict],
    reference_date: datetime,
    half_life_days: float = 180.0,
) -> float:
    if not matches:
        return 0.5

    weighted_wins = 0.0
    total_weight = 0.0
    for m in matches:
        age_days = (reference_date - m["played_at"]).total_seconds() / 86400.0
        if age_days < 0:
            continue
        weight = math.pow(0.5, age_days / half_life_days)
        total_weight += weight
        if m["team_won"]:
            weighted_wins += weight

    return weighted_wins / total_weight if total_weight > 0 else 0.5
