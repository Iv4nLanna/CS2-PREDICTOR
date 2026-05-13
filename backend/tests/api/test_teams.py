from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
)


def _seed_team_with_history(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", country="BR", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", country="EU", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=99, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now - timedelta(days=5),
        status=MatchStatus.FINISHED,
    )
    db_session.add(match)
    db_session.flush()
    db_session.add(MatchResult(
        match_id=match.id, winner_id=a.id,
        score_detail={"de_mirage": [16, 12]},
        played_at=now - timedelta(days=5),
    ))
    db_session.commit()
    return a


def test_list_teams(client, db_session):
    _seed_team_with_history(db_session)
    response = client.get("/teams")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert any(t["team"]["name"] == "A" for t in data)


def test_team_detail_returns_map_winrates(client, db_session):
    team = _seed_team_with_history(db_session)
    response = client.get(f"/teams/{team.id}/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["team"]["name"] == "A"
    assert "de_mirage" in data["map_winrates"]
    assert data["last_matches_played"] == 1


def test_team_detail_404_when_missing(client):
    response = client.get("/teams/99999/stats")
    assert response.status_code == 404
