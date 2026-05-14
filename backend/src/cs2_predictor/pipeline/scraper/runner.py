import logging

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from sqlalchemy.orm import Session

from cs2_predictor.pipeline.scraper.pipelines import set_session
from cs2_predictor.pipeline.scraper.spiders.match_detail import MatchDetailSpider
from cs2_predictor.pipeline.scraper.spiders.matches import UpcomingMatchesSpider
from cs2_predictor.pipeline.scraper.spiders.results import ResultsSpider
from cs2_predictor.pipeline.scraper.spiders.teams import TeamRankingSpider

logger = logging.getLogger(__name__)

SPIDERS = {
    "teams": TeamRankingSpider,
    "matches": UpcomingMatchesSpider,
    "results": ResultsSpider,
    "match_detail": MatchDetailSpider,
}


def run_scrapers(spider_names: list[str], session: Session | None = None,
                 match_ids: list[int] | None = None) -> dict:
    settings = get_project_settings()
    settings.set("ROBOTSTXT_OBEY", False)
    settings.set("CONCURRENT_REQUESTS", 4)
    settings.set("DOWNLOAD_DELAY", 0.5)
    settings.set("COOKIES_ENABLED", False)

    if session is not None:
        set_session(session)
        settings.set("ITEM_PIPELINES", {
            "cs2_predictor.pipeline.scraper.pipelines.TeamPipeline": 300,
        })

    process = CrawlerProcess(settings)
    for name in spider_names:
        if name not in SPIDERS:
            logger.warning("Unknown spider: %s", name)
            continue
        spider_cls = SPIDERS[name]
        if name == "match_detail":
            process.crawl(spider_cls, match_ids=match_ids or [])
        else:
            process.crawl(spider_cls)

    process.start()
    return {"status": "ok", "spiders": spider_names}
