from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cs2_predictor.api.schemas import (
    MatchDetail,
    MatchFeatureSet,
    MatchPrediction,
    TeamSummary,
)
from cs2_predictor.db.models import (
    Match,
    MatchStatus,
    Prediction,
    TeamFeatures,
)
from cs2_predictor.db.session import get_db

router = APIRouter(prefix="/matches", tags=["matches"])


def _to_team_summary(team) -> TeamSummary:
    return TeamSummary(
        id=team.id, hltv_id=team.hltv_id, name=team.name,
        country=team.country, hltv_ranking=team.hltv_ranking,
    )


def _to_prediction_dto(match: Match, prediction: Prediction) -> MatchPrediction:
    return MatchPrediction(
        match_id=match.id,
        hltv_match_id=match.hltv_id,
        team_a=_to_team_summary(match.team_a),
        team_b=_to_team_summary(match.team_b),
        team_a_win_prob=prediction.team_a_win_prob,
        team_b_win_prob=prediction.team_b_win_prob,
        format=match.format.value,
        is_lan=match.is_lan,
        scheduled_at=match.scheduled_at,
        tournament=match.tournament,
        model_version=prediction.model_version,
    )


def _latest_prediction_subquery(db: Session):
    from sqlalchemy import func

    return (
        db.query(
            Prediction.match_id.label("match_id"),
            func.max(Prediction.created_at).label("latest_created_at"),
        )
        .group_by(Prediction.match_id)
        .subquery()
    )


@router.get("/upcoming", response_model=list[MatchPrediction])
def list_upcoming(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    latest = _latest_prediction_subquery(db)
    rows = (
        db.query(Match, Prediction)
        .join(Prediction, Prediction.match_id == Match.id)
        .join(
            latest,
            (latest.c.match_id == Prediction.match_id)
            & (latest.c.latest_created_at == Prediction.created_at),
        )
        .filter(Match.status == MatchStatus.SCHEDULED)
        .filter(Match.scheduled_at >= now)
        .order_by(Match.scheduled_at.asc())
        .all()
    )
    return [_to_prediction_dto(m, p) for m, p in rows]
