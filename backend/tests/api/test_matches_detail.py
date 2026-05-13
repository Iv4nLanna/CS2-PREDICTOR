from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchStatus,
    ModelRun,
    Prediction,
    Team,
    TeamFeatures,
)


def _seed_full(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=False, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
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
        ModelRun(version="v1", trained_at=now, accuracy=0.7, features_used=["x"]),
        Prediction(match_id=match.id, team_a_win_prob=0.62,
                   team_b_win_prob=0.38, model_version="v1"),
    ])
    db_session.commit()
    return match


def test_prediction_endpoint_returns_full_detail(client, db_session):
    match = _seed_full(db_session)
    response = client.get(f"/matches/{match.id}/prediction")
    assert response.status_code == 200
    data = response.json()
    assert data["team_a_win_prob"] == 0.62
    assert data["tournament"] == "x"


def test_prediction_endpoint_404_when_missing(client):
    response = client.get("/matches/99999/prediction")
    assert response.status_code == 404


def test_features_endpoint_returns_both_teams(client, db_session):
    match = _seed_full(db_session)
    response = client.get(f"/matches/{match.id}/features")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    a_features = [f for f in data if f["win_rate_recent_decayed"] == 0.7][0]
    assert a_features["map_stats"] == {"de_mirage": 0.75}
