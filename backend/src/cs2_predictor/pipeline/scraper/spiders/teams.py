import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import TeamItem


class TeamRankingSpider(scrapy.Spider):
    name = "teams"
    start_urls = ["https://www.hltv.org/ranking/teams/"]

    def parse(self, response: Response):
        for rank_box in response.css(".ranked-team"):
            rank = rank_box.css(".position::text").re_first(r"\d+")
            name = rank_box.css(".teamName::text").get()
            country = rank_box.css(".teamName img::attr(title)").get()
            team_id = rank_box.css("a.moreLink::attr(href)").re_first(r"/team/(\d+)/")
            players = []
            for player_el in rank_box.css(".player-ratings .player"):
                player_name = player_el.css(".player-nick::text").get()
                player_rating = player_el.css(".rating::text").get()
                players.append({
                    "name": player_name,
                    "rating": float(player_rating) if player_rating else None,
                })
            item = TeamItem(
                hltv_id=int(team_id) if team_id else None,
                name=name,
                country=country,
                rank=int(rank) if rank else None,
                players=players,
            )
            yield item
