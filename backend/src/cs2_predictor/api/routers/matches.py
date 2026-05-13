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


@router.get("/{match_id}/prediction", response_model=MatchPrediction)
def get_prediction(match_id: int, db: Session = Depends(get_db)):
    row = (
        db.query(Match, Prediction)
        .join(Prediction, Prediction.match_id == Match.id)
        .filter(Match.id == match_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    match, prediction = row
    return _to_prediction_dto(match, prediction)


@router.get("/{match_id}/features", response_model=list[MatchFeatureSet])
def get_features(match_id: int, db: Session = Depends(get_db)):
    rows = db.query(TeamFeatures).filter_by(match_id=match_id).all()
    if not rows:
        raise HTTPException(status_code=404, detail="features not found")
    return [
        MatchFeatureSet(
            team_id=r.team_id,
            win_rate_recent_decayed=r.win_rate_recent_decayed,
            head_to_head_decayed=r.head_to_head_decayed,
            hltv_ranking_snapshot=r.hltv_ranking_snapshot,
            sos_score=r.sos_score,
            map_stats=r.map_stats,
        )
        for r in rows
    ]
