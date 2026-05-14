import logging

import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import MatchDetailItem

logger = logging.getLogger(__name__)


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip().rstrip("%"))
    except (ValueError, TypeError):
        return None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (ValueError, TypeError):
        return None


class MatchDetailSpider(scrapy.Spider):
    name = "match_detail"
    allowed_domains = ["hltv.org"]
    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,
    }

    def __init__(self, match_ids: list[int] | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_ids = match_ids or []

    def start_requests(self):
        for mid in self.match_ids:
            yield scrapy.Request(
                f"https://www.hltv.org/matches/{mid}/",
                cb_kwargs={"match_id": mid},
            )

    def parse(self, response: Response, match_id: int):
        team_stats = []
        for team_section in response.css(".stats-content"):
            team_name = team_section.css(".teamName::text").get()
            players = []
            for row in team_section.css("table tr"):
                name = row.css(".st-player a::text").get()
                if not name:
                    continue
                players.append({
                    "name": name.strip(),
                    "kills": _safe_int(row.css(".st-kills::text").get()),
                    "deaths": _safe_int(row.css(".st-deaths::text").get()),
                    "adr": _safe_float(row.css(".st-adr::text").get()),
                    "kast": _safe_float(row.css(".st-kast::text").get()),
                    "rating": _safe_float(row.css(".st-rating::text").get()),
                })
            if team_name:
                team_stats.append({"team_name": team_name.strip(), "players": players})

        if not team_stats:
            logger.warning("No stats found for match %d", match_id)

        yield MatchDetailItem(
            hltv_id=match_id,
            team_stats=team_stats,
        )
