from datetime import datetime, timedelta, timezone

import numpy as np

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchStatus,
    ModelRun,
    Prediction,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.model.predict import generate_predictions


class FakeCalibrated:
    feature_names = ["x"]
    base_accuracy = 0.7

    def predict_proba(self, X):
        return np.full(X.shape[0], 0.65)


def test_generates_prediction_per_scheduled_match(db_session):
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
                     hltv_ranking_snapshot=5, sos_score=0.6, map_stats={}),
        TeamFeatures(team_id=b.id, match_id=match.id,
                     win_rate_recent_decayed=0.4, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=20, sos_score=0.45, map_stats={}),
    ])
    db_session.add(ModelRun(version="v1", trained_at=now, accuracy=0.7,
                            features_used=["x"]))
    db_session.commit()

    count = generate_predictions(db_session, FakeCalibrated(), version="v1")
    db_session.commit()
    assert count == 1
    pred = db_session.query(Prediction).one()
    assert abs(pred.team_a_win_prob - 0.65) < 1e-6
    assert abs(pred.team_b_win_prob - 0.35) < 1e-6
    assert pred.model_version == "v1"


def test_skips_match_without_features(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.commit()

    count = generate_predictions(db_session, FakeCalibrated(), version="v1")
    db_session.commit()
    assert count == 0
    assert db_session.query(Prediction).count() == 0
