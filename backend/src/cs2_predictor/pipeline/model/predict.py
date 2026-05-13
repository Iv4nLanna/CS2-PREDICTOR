import numpy as np
from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchStatus,
    Prediction,
    TeamFeatures,
)
from cs2_predictor.pipeline.model.dataset import _feature_vector


def generate_predictions(session: Session, model, version: str) -> int:
    matches = session.query(Match).filter(Match.status == MatchStatus.SCHEDULED).all()
    count = 0
    for match in matches:
        tf_a = (
            session.query(TeamFeatures)
            .filter_by(team_id=match.team_a_id, match_id=match.id)
            .one_or_none()
        )
        tf_b = (
            session.query(TeamFeatures)
            .filter_by(team_id=match.team_b_id, match_id=match.id)
            .one_or_none()
        )
        if tf_a is None or tf_b is None:
            continue
        X = np.array([_feature_vector(tf_a, tf_b, match)], dtype=float)
        prob_a = float(model.predict_proba(X)[0])
        prob_b = 1.0 - prob_a

        existing = (
            session.query(Prediction)
            .filter_by(match_id=match.id, model_version=version)
            .one_or_none()
        )
        if existing is None:
            existing = Prediction(match_id=match.id, model_version=version)
            session.add(existing)
        existing.team_a_win_prob = prob_a
        existing.team_b_win_prob = prob_b
        existing.calibrated = True
        count += 1
    return count
