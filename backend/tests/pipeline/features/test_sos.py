from cs2_predictor.pipeline.features.sos import compute_sos


def test_no_matches_returns_zero():
    assert compute_sos([]) == 0.0


def test_wins_against_strong_opponents_score_high():
    matches = [
        {"opponent_rank": 5, "won": True},
        {"opponent_rank": 8, "won": True},
    ]
    weak = [
        {"opponent_rank": 100, "won": True},
        {"opponent_rank": 80, "won": True},
    ]
    assert compute_sos(matches) > compute_sos(weak)


def test_losses_count_against():
    wins = [{"opponent_rank": 10, "won": True}]
    losses = [{"opponent_rank": 10, "won": False}]
    assert compute_sos(wins) > compute_sos(losses)


def test_unranked_opponent_treated_as_weak():
    ranked = [{"opponent_rank": 5, "won": True}]
    unranked = [{"opponent_rank": None, "won": True}]
    assert compute_sos(ranked) > compute_sos(unranked)
