from cs2_predictor.pipeline.features.map_stats import compute_map_stats


def test_empty_returns_empty_dict():
    assert compute_map_stats([], map_pool=["de_mirage"]) == {"de_mirage": 0.5}


def test_single_map_winrate():
    matches = [
        {"map": "de_mirage", "won": True},
        {"map": "de_mirage", "won": True},
        {"map": "de_mirage", "won": False},
    ]
    result = compute_map_stats(matches, map_pool=["de_mirage"])
    assert abs(result["de_mirage"] - 2 / 3) < 1e-9


def test_only_maps_in_pool_returned():
    matches = [
        {"map": "de_mirage", "won": True},
        {"map": "de_inferno", "won": False},
    ]
    result = compute_map_stats(matches, map_pool=["de_mirage"])
    assert "de_inferno" not in result
    assert "de_mirage" in result


def test_pool_map_with_no_history_returns_neutral():
    result = compute_map_stats([], map_pool=["de_anubis", "de_nuke"])
    assert result == {"de_anubis": 0.5, "de_nuke": 0.5}
