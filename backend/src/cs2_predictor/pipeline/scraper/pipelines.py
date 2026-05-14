from cs2_predictor.db.models import Team
from cs2_predictor.pipeline.scraper.items import TeamItem

_SESSION = None


def set_session(session):
    global _SESSION
    _SESSION = session


class TeamPipeline:
    def process_item(self, item, spider):
        session = _SESSION
        if session is None:
            return item
        if isinstance(item, TeamItem):
            team = session.query(Team).filter_by(hltv_id=item["hltv_id"]).first()
            if not team:
                team = Team(hltv_id=item["hltv_id"])
                session.add(team)
            team.name = item.get("name") or ""
            team.country = item.get("country")
            team.hltv_ranking = item.get("rank")
        return item
