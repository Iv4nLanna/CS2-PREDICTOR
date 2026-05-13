from datetime import datetime, timezone

from cs2_predictor.db.models import Match, MatchFormat, MatchResult, MatchStatus, Team
from cs2_predictor.pipeline.scraper.persistence import (
    upsert_match_results,
    upsert_matches,
    upsert_teams,
)


def test_upsert_teams_inserts_new(db_session):
    payload = [
        {"id": 1, "name": "Navi", "country": "UA", "rank": 1},
        {"id": 2, "name": "FaZe", "country": "EU", "rank": 2},
    ]
    upsert_teams(db_session, payload)
    db_session.commit()
    assert db_session.query(Team).count() == 2


def test_upsert_teams_updates_existing(db_session):
    db_session.add(Team(hltv_id=1, name="Old", country="UA", hltv_ranking=5))
    db_session.commit()
    upsert_teams(db_session, [{"id": 1, "name": "Navi", "country": "UA", "rank": 1}])
    db_session.commit()
    team = db_session.query(Team).filter_by(hltv_id=1).one()
    assert team.name == "Navi"
    assert team.hltv_ranking == 1


def test_upsert_matches_creates_with_teams(db_session):
    db_session.add_all([Team(hltv_id=1, name="A"), Team(hltv_id=2, name="B")])
    db_session.commit()
    payload = [{
        "id": 100,
        "team_a_id": 1,
        "team_b_id": 2,
        "format": "BO3",
        "is_lan": True,
        "map_pool": ["de_mirage"],
        "tournament": "Major",
        "scheduled_at": "2026-06-01T18:00:00+00:00",
    }]
    upsert_matches(db_session, payload)
    db_session.commit()
    match = db_session.query(Match).filter_by(hltv_id=100).one()
    assert match.format == MatchFormat.BO3
    assert match.is_lan is True


def test_upsert_match_results_marks_finished(db_session):
    db_session.add_all([Team(hltv_id=1, name="A"), Team(hltv_id=2, name="B")])
    db_session.commit()
    upsert_matches(db_session, [{
        "id": 100, "team_a_id": 1, "team_b_id": 2, "format": "BO1",
        "is_lan": False, "map_pool": [], "tournament": "x",
        "scheduled_at": "2026-06-01T18:00:00+00:00",
    }])
    db_session.commit()
    upsert_match_results(db_session, [{
        "id": 100,
        "winner_id": 1,
        "score_detail": {"de_mirage": [16, 12]},
        "played_at": "2026-06-01T20:00:00+00:00",
    }])
    db_session.commit()
    match = db_session.query(Match).filter_by(hltv_id=100).one()
    assert match.status == MatchStatus.FINISHED
    result = db_session.query(MatchResult).filter_by(match_id=match.id).one()
    assert result.score_detail == {"de_mirage": [16, 12]}
