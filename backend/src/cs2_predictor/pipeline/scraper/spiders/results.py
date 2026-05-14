import re

import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import ResultItem


class ResultsSpider(scrapy.Spider):
    name = "results"
    allowed_domains = ["hltv.org"]
    start_urls = ["https://www.hltv.org/results"]
    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,
    }

    def parse(self, response: Response):
        for result_el in response.css(".result-con"):
            link = result_el.css("a.a-reset::attr(href)").get()
            match_id = None
            if link:
                ids = re.findall(r"/matches/(\d+)/", link)
                if ids:
                    match_id = int(ids[0])
            team_els = result_el.css(".team")
            teams = [t.css("::text").get() for t in team_els]
            scores = result_el.css(".result-score span::text").getall()
            event_name = result_el.css(".event-name::text").get()
            maps_played = result_el.css(".map-text::text").getall()

            if match_id and len(teams) >= 2:
                score_a = int(scores[0]) if len(scores) > 0 else 0
                score_b = int(scores[1]) if len(scores) > 1 else 0
                yield ResultItem(
                    hltv_id=match_id,
                    team_a_name=teams[0].strip() if teams[0] else None,
                    team_b_name=teams[1].strip() if teams[1] else None,
                    team_a_score=score_a,
                    team_b_score=score_b,
                    event_name=event_name.strip() if event_name else None,
                    maps=[m.strip() for m in maps_played if m.strip()],
                )

        next_page = response.css("a.pagination-next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
