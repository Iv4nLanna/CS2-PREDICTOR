import pytest

from cs2_predictor.db.models import Team
from cs2_predictor.pipeline.scraper.items import MatchItem, TeamItem
from cs2_predictor.pipeline.scraper.pipelines import TeamPipeline, set_session


def test_team_pipeline_creates_team(db_session):
    set_session(db_session)
    pipeline = TeamPipeline()
    item = TeamItem(hltv_id=42, name="NaVi", country="UA", rank=1, players=[])
    pipeline.process_item(item, None)
    db_session.flush()
    team = db_session.query(Team).filter_by(hltv_id=42).first()
    assert team is not None
    assert team.name == "NaVi"
    assert team.country == "UA"
    assert team.hltv_ranking == 1


def test_team_pipeline_updates_existing_team(db_session):
    set_session(db_session)
    existing = Team(hltv_id=42, name="OldName")
    db_session.add(existing)
    db_session.flush()
    pipeline = TeamPipeline()
    item = TeamItem(hltv_id=42, name="NewName", country="DE", rank=5, players=[])
    pipeline.process_item(item, None)
    db_session.flush()
    team = db_session.query(Team).filter_by(hltv_id=42).first()
    assert team.name == "NewName"
    assert team.country == "DE"
    assert team.hltv_ranking == 5


def test_team_pipeline_skips_non_team_item(db_session):
    set_session(db_session)
    pipeline = TeamPipeline()
    item = MatchItem(hltv_id=1, team_a_name="A", team_b_name="B", format="BO3")
    result = pipeline.process_item(item, None)
    assert result is item
    assert db_session.query(Team).count() == 0


def test_team_pipeline_no_session():
    set_session(None)
    pipeline = TeamPipeline()
    item = TeamItem(hltv_id=42, name="NaVi", country="UA", rank=1, players=[])
    result = pipeline.process_item(item, None)
    assert result is item


@pytest.fixture(autouse=True)
def reset_session():
    yield
    set_session(None)
