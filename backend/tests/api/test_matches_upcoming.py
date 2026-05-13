from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchStatus,
    ModelRun,
    Prediction,
    Team,
)


def _seed(db_session, with_prediction=True):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="Navi", country="UA", hltv_ranking=1)
    b = Team(hltv_id=2, name="FaZe", country="EU", hltv_ranking=2)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=True, map_pool=["de_mirage"],
        tournament="Major", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.flush()
    if with_prediction:
        db_session.add(ModelRun(version="v1", trained_at=now, accuracy=0.7,
                                features_used=["x"]))
        db_session.add(Prediction(match_id=match.id, team_a_win_prob=0.6,
                                  team_b_win_prob=0.4, model_version="v1"))
    db_session.commit()
    return match


def test_upcoming_returns_matches_with_predictions(client, db_session):
    _seed(db_session)
    response = client.get("/matches/upcoming")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["team_a"]["name"] == "Navi"
    assert data[0]["team_a_win_prob"] == 0.6
    assert data[0]["model_version"] == "v1"


def test_upcoming_omits_matches_without_prediction(client, db_session):
    _seed(db_session, with_prediction=False)
    response = client.get("/matches/upcoming")
    assert response.status_code == 200
    assert response.json() == []
