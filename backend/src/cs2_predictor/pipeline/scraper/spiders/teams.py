import logging

import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import TeamItem
from cs2_predictor.pipeline.scraper.utils import safe_float

logger = logging.getLogger(__name__)


class TeamRankingSpider(scrapy.Spider):
    name = "teams"
    allowed_domains = ["hltv.org"]
    start_urls = ["https://www.hltv.org/ranking/teams/"]

    def parse(self, response: Response):
        for rank_box in response.css(".ranked-team"):
            rank = rank_box.css(".position::text").re_first(r"\d+")
            name = rank_box.css(".teamName::text").get()
            country = rank_box.css(".teamName img::attr(title)").get()
            team_id = rank_box.css("a.moreLink::attr(href)").re_first(r"/team/(\d+)/")
            if not name:
                logger.warning("Team without name in ranking box")
                continue
            players = []
            for player_el in rank_box.css(".player-ratings .player"):
                player_name = player_el.css(".player-nick::text").get()
                player_rating = player_el.css(".rating::text").get()
                players.append({
                    "name": player_name,
                    "rating": safe_float(player_rating),
                })
            yield TeamItem(
                hltv_id=int(team_id) if team_id else None,
                name=name.strip(),
                country=country,
                rank=int(rank) if rank else None,
                players=players,
            )
