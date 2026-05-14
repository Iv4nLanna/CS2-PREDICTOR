from cs2_predictor.pipeline.scraper.items import MatchDetailItem, MatchItem, ResultItem, TeamItem


def test_items_import():
    assert TeamItem is not None
    assert MatchItem is not None
    assert ResultItem is not None
    assert MatchDetailItem is not None


def test_team_item_fields():
    item = TeamItem(hltv_id=1, name="NaVi", country="UA", rank=1, players=[])
    assert item["hltv_id"] == 1
    assert item["name"] == "NaVi"


def test_match_item_fields():
    item = MatchItem(hltv_id=100, team_a_name="A", team_b_name="B", format="BO3")
    assert item["hltv_id"] == 100
    assert item["format"] == "BO3"


def test_team_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.teams import TeamRankingSpider
    spider = TeamRankingSpider()
    assert spider.name == "teams"
    assert "hltv.org/ranking" in spider.start_urls[0]


def test_parse_ranking_safe_rating():
    from cs2_predictor.pipeline.scraper.spiders.teams import TeamRankingSpider
    spider = TeamRankingSpider()
    html = """
    <div class="ranked-team">
      <span class="position">1</span>
      <span class="teamName"><img title="Brazil">Team X</span>
      <a class="moreLink" href="/team/123/test"></a>
      <div class="player-ratings">
        <div class="player">
          <span class="player-nick">player1</span>
          <span class="rating">1.15</span>
        </div>
        <div class="player">
          <span class="player-nick">player2</span>
          <span class="rating">-</span>
        </div>
      </div>
    </div>
    """
    from scrapy.http import HtmlResponse
    response = HtmlResponse(url="http://example.com", body=html, encoding="utf-8")
    items = list(spider.parse(response))
    assert len(items) == 1
    assert items[0]["hltv_id"] == 123
    assert items[0]["players"][0]["rating"] == 1.15
    assert items[0]["players"][1]["rating"] is None


def test_matches_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.matches import UpcomingMatchesSpider
    spider = UpcomingMatchesSpider()
    assert spider.name == "matches"
    assert "hltv.org/matches" in spider.start_urls[0]


def test_results_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.results import ResultsSpider
    spider = ResultsSpider()
    assert spider.name == "results"
    assert spider.start_urls == ["https://www.hltv.org/results"]


def test_match_detail_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.match_detail import MatchDetailSpider
    spider = MatchDetailSpider(match_ids=[1, 2])
    assert spider.name == "match_detail"
    assert spider.match_ids == [1, 2]


def test_runner_import():
    from cs2_predictor.pipeline.scraper.runner import run_scrapers
    assert callable(run_scrapers)


def test_safe_float():
    from cs2_predictor.pipeline.scraper.utils import safe_float, safe_int
    assert safe_float("1.15") == 1.15
    assert safe_float("-") is None
    assert safe_float("") is None
    assert safe_float(None) is None
    assert safe_float("85.7%") == 85.7
    assert safe_int("10") == 10
    assert safe_int("abc") is None
    assert safe_int(None) is None
