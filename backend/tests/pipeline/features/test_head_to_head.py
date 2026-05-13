from datetime import datetime, timedelta, timezone

from cs2_predictor.pipeline.features.head_to_head import compute_head_to_head


def test_no_history_returns_neutral():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert compute_head_to_head([], reference_date=ref) == 0.5


def test_perfect_record_recent():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [
        {"played_at": ref - timedelta(days=10), "team_won": True}
        for _ in range(3)
    ]
    score = compute_head_to_head(matches, reference_date=ref)
    assert score > 0.9


def test_old_match_weighs_less_than_recent():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    recent = [{"played_at": ref - timedelta(days=10), "team_won": True}]
    old = [{"played_at": ref - timedelta(days=400), "team_won": True}]
    assert compute_head_to_head(recent, reference_date=ref) > compute_head_to_head(old, reference_date=ref)
