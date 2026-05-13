from datetime import datetime, timedelta, timezone

from cs2_predictor.pipeline.features.recent_form import compute_recent_form


def _match(date, won):
    return {"played_at": date, "won": won}


def test_no_matches_returns_neutral():
    assert compute_recent_form([], reference_date=datetime.now(timezone.utc)) == 0.5


def test_all_wins_recent_returns_close_to_one():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [_match(ref - timedelta(days=i), won=True) for i in range(1, 6)]
    score = compute_recent_form(matches, reference_date=ref, half_life_days=30)
    assert score > 0.9


def test_all_losses_returns_close_to_zero():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [_match(ref - timedelta(days=i), won=False) for i in range(1, 6)]
    score = compute_recent_form(matches, reference_date=ref, half_life_days=30)
    assert score < 0.1


def test_recent_match_weighs_more_than_old():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    recent_win = [_match(ref - timedelta(days=1), True)]
    old_win = [_match(ref - timedelta(days=180), True)]
    recent_score = compute_recent_form(recent_win, reference_date=ref, half_life_days=30)
    old_score = compute_recent_form(old_win, reference_date=ref, half_life_days=30)
    assert recent_score > old_score


def test_mix_returns_between_zero_and_one():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [
        _match(ref - timedelta(days=1), True),
        _match(ref - timedelta(days=2), False),
        _match(ref - timedelta(days=3), True),
    ]
    score = compute_recent_form(matches, reference_date=ref)
    assert 0.0 < score < 1.0
