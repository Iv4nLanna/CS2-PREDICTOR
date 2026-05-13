UNRANKED_PROXY = 150


def _opponent_weight(rank: int | None) -> float:
    effective = rank if rank is not None else UNRANKED_PROXY
    effective = max(effective, 1)
    return 1.0 / (1.0 + 0.05 * (effective - 1))


def compute_sos(matches: list[dict]) -> float:
    if not matches:
        return 0.0

    total = 0.0
    for m in matches:
        w = _opponent_weight(m.get("opponent_rank"))
        total += w if m["won"] else -w

    return total / len(matches)
