import math
from datetime import datetime


def compute_recent_form(
    matches: list[dict],
    reference_date: datetime,
    half_life_days: float = 30.0,
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
        if m["won"]:
            weighted_wins += weight

    if total_weight == 0:
        return 0.5
    prior_weight = 1.0
    return (weighted_wins + prior_weight * 0.5) / (total_weight + prior_weight)
