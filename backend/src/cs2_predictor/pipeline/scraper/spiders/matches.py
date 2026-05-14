import re
from datetime import datetime

import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import MatchItem


def _parse_format(text: str) -> str:
    text = (text or "").lower()
    if "bo5" in text:
        return "BO5"
    if "bo3" in text:
        return "BO3"
    return "BO1"


def _is_lan(text: str) -> bool:
    return "lan" in (text or "").lower()


class UpcomingMatchesSpider(scrapy.Spider):
    name = "matches"
    allowed_domains = ["hltv.org"]
    start_urls = ["https://www.hltv.org/matches"]

    def parse(self, response: Response):
        for match_el in response.css(".upcoming-match"):
            urls = match_el.css("a::attr(href)").getall()
            match_id = None
            for url in urls:
                ids = re.findall(r"/matches/(\d+)/", url)
                if ids:
                    match_id = int(ids[0])
                    break
            teams = match_el.css(".matchTeamName::text").getall()
            team_names = [t.strip() for t in teams if t.strip()]
            format_text = match_el.css(".matchMeta::text").get()
            event_name = match_el.css(".matchEventName::text").get()
            maps = match_el.css(".mapname::text").getall()
            time_attr = match_el.css(".matchTime::attr(data-unix)").get()

            if len(team_names) >= 2 and match_id:
                scheduled_at = None
                if time_attr:
                    scheduled_at = datetime.fromtimestamp(int(time_attr) / 1000)
                yield MatchItem(
                    hltv_id=match_id,
                    team_a_name=team_names[0],
                    team_b_name=team_names[1],
                    format=_parse_format(format_text),
                    is_lan=_is_lan(format_text),
                    event_name=event_name.strip() if event_name else None,
                    scheduled_at=scheduled_at,
                    map_pool=maps,
                )
