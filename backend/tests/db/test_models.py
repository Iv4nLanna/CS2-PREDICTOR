from datetime import datetime, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    ModelRun,
    Player,
    Prediction,
    Team,
    TeamFeatures,
)


def test_team_persists(db_session):
    team = Team(hltv_id=4608, name="Natus Vincere", country="Ukraine", hltv_ranking=1)
    db_session.add(team)
    db_session.commit()
    fetched = db_session.query(Team).filter_by(hltv_id=4608).one()
    assert fetched.name == "Natus Vincere"


def test_match_links_two_teams(db_session):
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100,
        team_a_id=a.id,
        team_b_id=b.id,
        format=MatchFormat.BO3,
        is_lan=True,
        map_pool=["de_mirage", "de_inferno", "de_anubis"],
        tournament="Major",
        scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.commit()
    fetched = db_session.query(Match).filter_by(hltv_id=100).one()
    assert fetched.team_a.name == "A"
    assert fetched.team_b.name == "B"
    assert fetched.map_pool == ["de_mirage", "de_inferno", "de_anubis"]


def test_prediction_links_to_match_and_model_run(db_session):
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.SCHEDULED,
    )
    run = ModelRun(version="v1", trained_at=datetime.now(timezone.utc),
                   accuracy=0.65, features_used=["win_rate_recent_decayed"])
    db_session.add_all([match, run])
    db_session.flush()
    pred = Prediction(match_id=match.id, team_a_win_prob=0.6,
                      team_b_win_prob=0.4, model_version="v1", calibrated=True)
    db_session.add(pred)
    db_session.commit()
    fetched = db_session.query(Prediction).one()
    assert fetched.team_a_win_prob == 0.6
    assert fetched.match.hltv_id == 100


def test_player_optional_faceit_columns(db_session):
    team = Team(hltv_id=1, name="A")
    db_session.add(team)
    db_session.flush()
    player = Player(team_id=team.id, hltv_id=999, name="s1mple",
                    rating=1.20, role="rifler", active=True)
    db_session.add(player)
    db_session.commit()
    fetched = db_session.query(Player).filter_by(hltv_id=999).one()
    assert fetched.faceit_id is None
    assert fetched.faceit_rating is None


def test_team_features_per_match(db_session):
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.flush()
    tf = TeamFeatures(
        team_id=a.id, match_id=match.id,
        win_rate_recent_decayed=0.7,
        head_to_head_decayed=0.5,
        hltv_ranking_snapshot=1,
        sos_score=0.8,
        map_stats={"de_mirage": 0.75},
    )
    db_session.add(tf)
    db_session.commit()
    fetched = db_session.query(TeamFeatures).one()
    assert fetched.map_stats["de_mirage"] == 0.75


def test_match_result_unique_per_match(db_session):
    from sqlalchemy.exc import IntegrityError

    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.FINISHED,
    )
    db_session.add(match)
    db_session.flush()
    r1 = MatchResult(match_id=match.id, winner_id=a.id,
                     score_detail={"de_mirage": [16, 12]},
                     played_at=datetime.now(timezone.utc))
    db_session.add(r1)
    db_session.commit()
    r2 = MatchResult(match_id=match.id, winner_id=b.id,
                     score_detail={}, played_at=datetime.now(timezone.utc))
    db_session.add(r2)
    try:
        db_session.commit()
        raise AssertionError("expected IntegrityError")
    except IntegrityError:
        db_session.rollback()
