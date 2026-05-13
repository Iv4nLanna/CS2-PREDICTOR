from datetime import datetime, timezone

from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def upsert_teams(session: Session, payload: list[dict]) -> None:
    for item in payload:
        team = session.query(Team).filter_by(hltv_id=item["id"]).one_or_none()
        if team is None:
            team = Team(hltv_id=item["id"])
            session.add(team)
        team.name = item.get("name", team.name if team.id else "")
        team.country = item.get("country")
        team.hltv_ranking = item.get("rank")
        team.updated_at = datetime.now(timezone.utc)


def upsert_matches(session: Session, payload: list[dict]) -> None:
    for item in payload:
        team_a = session.query(Team).filter_by(hltv_id=item["team_a_id"]).one()
        team_b = session.query(Team).filter_by(hltv_id=item["team_b_id"]).one()
        match = session.query(Match).filter_by(hltv_id=item["id"]).one_or_none()
        if match is None:
            match = Match(hltv_id=item["id"])
            session.add(match)
        match.team_a_id = team_a.id
        match.team_b_id = team_b.id
        match.format = MatchFormat(item["format"])
        match.is_lan = bool(item.get("is_lan", False))
        match.map_pool = item.get("map_pool", [])
        match.tournament = item.get("tournament")
        match.scheduled_at = _parse_dt(item["scheduled_at"])
        if match.status is None:
            match.status = MatchStatus.SCHEDULED


def upsert_match_results(session: Session, payload: list[dict]) -> None:
    for item in payload:
        match = session.query(Match).filter_by(hltv_id=item["id"]).one()
        winner = session.query(Team).filter_by(hltv_id=item["winner_id"]).one()
        result = session.query(MatchResult).filter_by(match_id=match.id).one_or_none()
        if result is None:
            result = MatchResult(match_id=match.id)
            session.add(result)
        result.winner_id = winner.id
        result.score_detail = item.get("score_detail", {})
        result.played_at = _parse_dt(item["played_at"])
        match.status = MatchStatus.FINISHED
