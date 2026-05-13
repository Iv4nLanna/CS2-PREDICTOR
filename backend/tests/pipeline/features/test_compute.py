from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.features.compute import compute_features_for_scheduled_matches


def _seed(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", hltv_ranking=20)
    c = Team(hltv_id=3, name="C", hltv_ranking=50)
    db_session.add_all([a, b, c])
    db_session.flush()
    past = Match(
        hltv_id=10, team_a_id=a.id, team_b_id=c.id, format=MatchFormat.BO1,
        is_lan=False, map_pool=["de_mirage"], tournament="x",
        scheduled_at=now - timedelta(days=20), status=MatchStatus.FINISHED,
    )
    db_session.add(past)
    db_session.flush()
    db_session.add(MatchResult(
        match_id=past.id, winner_id=a.id,
        score_detail={"de_mirage": [16, 10]},
        played_at=now - timedelta(days=20),
    ))
    upcoming = Match(
        hltv_id=11, team_a_id=a.id, team_b_id=b.id, format=MatchFormat.BO3,
        is_lan=True, map_pool=["de_mirage", "de_inferno"], tournament="x",
        scheduled_at=now + timedelta(days=2), status=MatchStatus.SCHEDULED,
    )
    db_session.add(upcoming)
    db_session.commit()
    return a, b, upcoming


def test_creates_two_team_feature_rows_per_match(db_session):
    a, b, upcoming = _seed(db_session)
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    rows = db_session.query(TeamFeatures).filter_by(match_id=upcoming.id).all()
    assert len(rows) == 2
    teams = {r.team_id for r in rows}
    assert teams == {a.id, b.id}


def test_features_have_expected_ranges(db_session):
    a, b, upcoming = _seed(db_session)
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    row_a = db_session.query(TeamFeatures).filter_by(team_id=a.id, match_id=upcoming.id).one()
    assert 0.0 <= row_a.win_rate_recent_decayed <= 1.0
    assert 0.0 <= row_a.head_to_head_decayed <= 1.0
    assert 0.0 <= row_a.sos_score <= 1.0
    assert "de_mirage" in row_a.map_stats
    assert row_a.hltv_ranking_snapshot == 5


def test_idempotent_recompute(db_session):
    _seed(db_session)
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    assert db_session.query(TeamFeatures).count() == 2
