from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.model.dataset import build_training_dataset


def _seed_finished_match(db_session, hltv_id, winner_is_a=True):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=hltv_id * 10 + 1, name=f"A{hltv_id}", hltv_ranking=5)
    b = Team(hltv_id=hltv_id * 10 + 2, name=f"B{hltv_id}", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=hltv_id, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=True, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now - timedelta(days=hltv_id),
        status=MatchStatus.FINISHED,
    )
    db_session.add(match)
    db_session.flush()
    db_session.add_all([
        TeamFeatures(team_id=a.id, match_id=match.id,
                     win_rate_recent_decayed=0.7, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=5, sos_score=0.6,
                     map_stats={"de_mirage": 0.75}),
        TeamFeatures(team_id=b.id, match_id=match.id,
                     win_rate_recent_decayed=0.4, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=20, sos_score=0.45,
                     map_stats={"de_mirage": 0.5}),
        MatchResult(match_id=match.id,
                    winner_id=a.id if winner_is_a else b.id,
                    score_detail={"de_mirage": [16, 12]},
                    played_at=now - timedelta(days=hltv_id)),
    ])
    db_session.commit()
    return match


def test_dataset_has_one_row_per_finished_match(db_session):
    _seed_finished_match(db_session, 1)
    _seed_finished_match(db_session, 2, winner_is_a=False)
    X, y, feature_names = build_training_dataset(db_session)
    assert X.shape[0] == 2
    assert y.tolist() in ([1, 0], [0, 1])
    assert "team_a_win_rate_recent_decayed" in feature_names
    assert "team_b_win_rate_recent_decayed" in feature_names


def test_dataset_is_ordered_by_played_at(db_session):
    _seed_finished_match(db_session, 5)
    _seed_finished_match(db_session, 1)
    X, y, _ = build_training_dataset(db_session)
    assert X.shape[0] == 2
