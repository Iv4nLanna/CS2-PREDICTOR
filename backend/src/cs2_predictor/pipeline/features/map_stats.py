from collections import defaultdict


def compute_map_stats(matches: list[dict], map_pool: list[str]) -> dict[str, float]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for m in matches:
        bucket = counts[m["map"]]
        bucket[1] += 1
        if m["won"]:
            bucket[0] += 1

    result: dict[str, float] = {}
    for map_name in map_pool:
        wins, total = counts.get(map_name, (0, 0))
        result[map_name] = wins / total if total > 0 else 0.5
    return result
