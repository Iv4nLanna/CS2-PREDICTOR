import numpy as np
from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchResult,
    MatchStatus,
    TeamFeatures,
)

SCALAR_FEATURES = [
    "win_rate_recent_decayed",
    "head_to_head_decayed",
    "hltv_ranking_snapshot",
    "sos_score",
]
CONTEXT_FEATURES = ["is_lan", "format_bo1", "format_bo3", "format_bo5"]


def _feature_vector(tf_a: TeamFeatures, tf_b: TeamFeatures, match: Match) -> list[float]:
    row: list[float] = []
    for name in SCALAR_FEATURES:
        row.append(float(getattr(tf_a, name) or 0.0))
    for name in SCALAR_FEATURES:
        row.append(float(getattr(tf_b, name) or 0.0))
    row.append(1.0 if match.is_lan else 0.0)
    row.append(1.0 if match.format.value == "BO1" else 0.0)
    row.append(1.0 if match.format.value == "BO3" else 0.0)
    row.append(1.0 if match.format.value == "BO5" else 0.0)
    return row


def feature_names() -> list[str]:
    names = [f"team_a_{n}" for n in SCALAR_FEATURES]
    names += [f"team_b_{n}" for n in SCALAR_FEATURES]
    names += CONTEXT_FEATURES
    return names


def build_training_dataset(session: Session) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = (
        session.query(Match, MatchResult)
        .join(MatchResult, MatchResult.match_id == Match.id)
        .filter(Match.status == MatchStatus.FINISHED)
        .order_by(MatchResult.played_at.asc())
        .all()
    )
    X_list, y_list = [], []
    for match, result in rows:
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
        X_list.append(_feature_vector(tf_a, tf_b, match))
        y_list.append(1 if result.winner_id == match.team_a_id else 0)

    X = np.array(X_list, dtype=float) if X_list else np.empty((0, len(feature_names())))
    y = np.array(y_list, dtype=int)
    return X, y, feature_names()
