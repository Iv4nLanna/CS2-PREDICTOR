from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchResult,
    MatchStatus,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.features.head_to_head import compute_head_to_head
from cs2_predictor.pipeline.features.map_stats import compute_map_stats
from cs2_predictor.pipeline.features.recent_form import compute_recent_form
from cs2_predictor.pipeline.features.sos import compute_sos


def _finished_matches_for_team(session: Session, team_id: int, before: datetime) -> list[dict]:
    stmt = (
        select(Match, MatchResult)
        .join(MatchResult, MatchResult.match_id == Match.id)
        .where(Match.status == MatchStatus.FINISHED)
        .where(or_(Match.team_a_id == team_id, Match.team_b_id == team_id))
        .where(MatchResult.played_at < before)
    )
    rows = session.execute(stmt).all()
    output = []
    for match, result in rows:
        opponent_id = match.team_b_id if match.team_a_id == team_id else match.team_a_id
        opponent = session.get(Team, opponent_id)
        opponent_rank = opponent.hltv_ranking if opponent else None
        for map_name, _score in (result.score_detail or {}).items():
            output.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": map_name,
                "opponent_rank": opponent_rank,
                "opponent_id": opponent_id,
            })
        if not (result.score_detail or {}):
            output.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": None,
                "opponent_rank": opponent_rank,
                "opponent_id": opponent_id,
            })
    return output


def _team_features(
    session: Session, team_id: int, opponent_id: int, match: Match
) -> TeamFeatures:
    ref = match.scheduled_at
    history = _finished_matches_for_team(session, team_id, before=ref)
    h2h = [m for m in history if m["opponent_id"] == opponent_id]
    h2h_dicts = [{"played_at": m["played_at"], "team_won": m["won"]} for m in h2h]
    map_matches = [{"map": m["map"], "won": m["won"]} for m in history if m["map"]]

    team = session.get(Team, team_id)
    existing = (
        session.query(TeamFeatures)
        .filter_by(team_id=team_id, match_id=match.id)
        .one_or_none()
    )
    if existing is None:
        existing = TeamFeatures(team_id=team_id, match_id=match.id)
        session.add(existing)

    existing.win_rate_recent_decayed = compute_recent_form(
        [{"played_at": m["played_at"], "won": m["won"]} for m in history],
        reference_date=ref,
    )
    existing.head_to_head_decayed = compute_head_to_head(h2h_dicts, reference_date=ref)
    existing.hltv_ranking_snapshot = team.hltv_ranking if team else None
    existing.sos_score = compute_sos(
        [{"opponent_rank": m["opponent_rank"], "won": m["won"]} for m in history]
    )
    existing.map_stats = compute_map_stats(map_matches, map_pool=match.map_pool or [])
    existing.computed_at = datetime.now(timezone.utc)
    return existing


def compute_features_for_scheduled_matches(session: Session) -> int:
    matches = session.query(Match).filter(Match.status == MatchStatus.SCHEDULED).all()
    count = 0
    for match in matches:
        _team_features(session, match.team_a_id, match.team_b_id, match)
        _team_features(session, match.team_b_id, match.team_a_id, match)
        count += 1
    return count
