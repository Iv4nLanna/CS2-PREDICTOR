from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from cs2_predictor.api.schemas import TeamDetail, TeamStats, TeamSummary
from cs2_predictor.db.models import (
    Match,
    MatchResult,
    MatchStatus,
    Team,
)
from cs2_predictor.db.session import get_db
from cs2_predictor.pipeline.features.map_stats import compute_map_stats
from cs2_predictor.pipeline.features.recent_form import compute_recent_form

router = APIRouter(prefix="/teams", tags=["teams"])


def _team_history(db: Session, team_id: int) -> list[dict]:
    rows = (
        db.query(Match, MatchResult)
        .join(MatchResult, MatchResult.match_id == Match.id)
        .filter(Match.status == MatchStatus.FINISHED)
        .filter(or_(Match.team_a_id == team_id, Match.team_b_id == team_id))
        .all()
    )
    history = []
    for match, result in rows:
        for map_name, _score in (result.score_detail or {}).items():
            history.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": map_name,
            })
        if not (result.score_detail or {}):
            history.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": None,
            })
    return history


def _to_summary(team: Team) -> TeamSummary:
    return TeamSummary(
        id=team.id, hltv_id=team.hltv_id, name=team.name,
        country=team.country, hltv_ranking=team.hltv_ranking,
    )


@router.get("", response_model=list[TeamStats])
def list_teams(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    teams = db.query(Team).order_by(Team.hltv_ranking.asc().nulls_last()).all()
    out: list[TeamStats] = []
    for team in teams:
        history = _team_history(db, team.id)
        out.append(TeamStats(
            team=_to_summary(team),
            recent_form=compute_recent_form(
                [{"played_at": h["played_at"], "won": h["won"]} for h in history],
                reference_date=now,
            ),
            last_matches_played=len({(h["played_at"]) for h in history}),
        ))
    return out


@router.get("/{team_id}/stats", response_model=TeamDetail)
def team_stats(team_id: int, db: Session = Depends(get_db)):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="team not found")
    history = _team_history(db, team.id)
    map_history = [{"map": h["map"], "won": h["won"]} for h in history if h["map"]]
    all_maps = sorted({m["map"] for m in map_history})
    return TeamDetail(
        team=_to_summary(team),
        recent_form=compute_recent_form(
            [{"played_at": h["played_at"], "won": h["won"]} for h in history],
            reference_date=datetime.now(timezone.utc),
        ),
        map_winrates=compute_map_stats(map_history, map_pool=all_maps),
        last_matches_played=len({h["played_at"] for h in history}),
    )
