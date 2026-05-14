# Scrapy HLTV Scraper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Before implementing any feature/bugfix, invoke `superpowers:test-driven-development`. For any failing test or unexpected error, invoke `superpowers:systematic-debugging`. Before claiming a task complete, invoke `superpowers:verification-before-completion`.

**Goal:** Substituir o scraper baseado em `eupeutro/hltv-api` por um scraper Scrapy que extrai dados diretamente do HLTV.org, incluindo stats de jogadores.

**Architecture:** Scrapy spiders extraem dados do HLTV em 4 spiders (teams, matches, results, match_detail). Items passam por pipelines que persistem via SQLAlchemy. `runner.py` expõe `run_scrapers()` que o pipeline chama.

**Tech Stack:** Python 3.12, Scrapy 2.11+, SQLAlchemy 2.x, pytest

**Spec reference:** `docs/superpowers/specs/2026-05-14-scrapy-hltv-scraper-design.md`

---

### Task 1: Instalar Scrapy e criar estrutura de spiders

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `backend/src/cs2_predictor/pipeline/scraper/spiders/__init__.py`
- Create: `backend/src/cs2_predictor/pipeline/scraper/items.py`
- Create: `backend/src/cs2_predictor/pipeline/scraper/middleware.py`
- Delete: `backend/src/cs2_predictor/pipeline/scraper/hltv.py`
- Delete: `backend/tests/pipeline/test_hltv_scraper.py`
- Test: `backend/tests/pipeline/test_spiders.py`

- [ ] **Step 1: Adicionar Scrapy às dependências**

Editar `backend/pyproject.toml`, adicionar no `[project] dependencies`:
```
    "scrapy>=2.11",
```

- [ ] **Step 2: Instalar dependência**

Run: `cd backend && uv pip install -e ".[dev]"`
Expected: scrapy instalado sem erro.

- [ ] **Step 3: Criar diretórios e arquivos vazios**

```bash
mkdir -p backend/src/cs2_predictor/pipeline/scraper/spiders
touch backend/src/cs2_predictor/pipeline/scraper/spiders/__init__.py
touch backend/src/cs2_predictor/pipeline/scraper/__init__.py
touch backend/src/cs2_predictor/pipeline/scraper/items.py
touch backend/src/cs2_predictor/pipeline/scraper/middleware.py
```

- [ ] **Step 4: Criar `backend/src/cs2_predictor/pipeline/scraper/items.py`**

```python
import scrapy
from scrapy import Field, Item


class TeamItem(Item):
    hltv_id = Field()
    name = Field()
    country = Field()
    rank = Field()
    players = Field()


class MatchItem(Item):
    hltv_id = Field()
    team_a_id = Field()
    team_b_id = Field()
    team_a_name = Field()
    team_b_name = Field()
    format = Field()
    is_lan = Field()
    event_name = Field()
    scheduled_at = Field()
    map_pool = Field()


class ResultItem(Item):
    hltv_id = Field()
    team_a_id = Field()
    team_b_id = Field()
    team_a_name = Field()
    team_b_name = Field()
    team_a_score = Field()
    team_b_score = Field()
    format = Field()
    event_name = Field()
    played_at = Field()
    maps = Field()


class MatchDetailItem(Item):
    hltv_id = Field()
    team_stats = Field()
    map_results = Field()
```

- [ ] **Step 5: Criar `backend/src/cs2_predictor/pipeline/scraper/middleware.py`**

```python
from scrapy import signals


class HLTVSpiderMiddleware:
    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        request.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        return None

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)
```

- [ ] **Step 6: Escrever teste de importação**

Criar `backend/tests/pipeline/test_spiders.py`:

```python
from cs2_predictor.pipeline.scraper.items import TeamItem, MatchItem, ResultItem, MatchDetailItem


def test_items_import():
    assert TeamItem is not None
    assert MatchItem is not None
    assert ResultItem is not None
    assert MatchDetailItem is not None
```

- [ ] **Step 7: Rodar teste**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: 1 PASSED

- [ ] **Step 8: Remover scraper antigo**

```bash
rm backend/src/cs2_predictor/pipeline/scraper/hltv.py
rm backend/tests/pipeline/test_hltv_scraper.py
```

- [ ] **Step 9: Commit**

```bash
git add backend/pyproject.toml backend/src/cs2_predictor/pipeline/scraper/ backend/tests/pipeline/test_spiders.py
git rm backend/src/cs2_predictor/pipeline/scraper/hltv.py backend/tests/pipeline/test_hltv_scraper.py
git commit -m "feat(scraper): add Scrapy items, middleware, and project structure"
```

---

### Task 2: Spider de Teams (ranking + perfil)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/scraper/spiders/teams.py`
- Test: `backend/tests/pipeline/test_spiders.py` (adicionar)

- [ ] **Step 1: Escrever teste que falha**

Adicionar em `backend/tests/pipeline/test_spiders.py`:

```python
def test_team_spider_parse_ranking():
    from cs2_predictor.pipeline.scraper.spiders.teams import TeamRankingSpider
    spider = TeamRankingSpider()
    assert spider.name == "teams"
    assert spider.start_urls == ["https://www.hltv.org/ranking/teams/"]
```

- [ ] **Step 2: Rodar teste**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/scraper/spiders/teams.py`**

```python
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
```

- [ ] **Step 4: Rodar testes — devem passar (teste de importação)**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/scraper/spiders/teams.py backend/tests/pipeline/test_spiders.py
git commit -m "feat(scraper): add team ranking spider"
```

---

### Task 3: Spider de Matches (partidas agendadas)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/scraper/spiders/matches.py`
- Modify: `backend/tests/pipeline/test_spiders.py`

- [ ] **Step 1: Escrever teste**

Adicionar em `backend/tests/pipeline/test_spiders.py`:

```python
def test_matches_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.matches import UpcomingMatchesSpider
    spider = UpcomingMatchesSpider()
    assert spider.name == "matches"
    assert "hltv.org/matches" in spider.start_urls[0]
```

- [ ] **Step 2: Rodar teste**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py::test_matches_spider_import -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/scraper/spiders/matches.py`**

```python
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
```

- [ ] **Step 4: Rodar testes**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/scraper/spiders/matches.py backend/tests/pipeline/test_spiders.py
git commit -m "feat(scraper): add upcoming matches spider"
```

---

### Task 4: Spider de Results (resultados históricos paginados)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/scraper/spiders/results.py`
- Modify: `backend/tests/pipeline/test_spiders.py`

- [ ] **Step 1: Escrever teste**

Adicionar em `backend/tests/pipeline/test_spiders.py`:

```python
def test_results_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.results import ResultsSpider
    spider = ResultsSpider()
    assert spider.name == "results"
    assert spider.start_urls == ["https://www.hltv.org/results"]
```

- [ ] **Step 2: Rodar teste**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py::test_results_spider_import -v`
Expected: FAIL

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/scraper/spiders/results.py`**

```python
import re
from datetime import datetime

import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import ResultItem


class ResultsSpider(scrapy.Spider):
    name = "results"
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
            stars = result_el.css(".star i::attr(class)").getall()

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

        # Paginate
        next_page = response.css("a.pagination-next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
```

- [ ] **Step 4: Rodar testes**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/scraper/spiders/results.py backend/tests/pipeline/test_spiders.py
git commit -m "feat(scraper): add results spider with pagination"
```

---

### Task 5: Spider de Match Detail (stats dos jogadores)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/scraper/spiders/match_detail.py`
- Modify: `backend/tests/pipeline/test_spiders.py`
- Create: `backend/tests/pipeline/test_spiders_data/` (dados estáticos para teste)

- [ ] **Step 1: Escrever teste de importação**

Adicionar em `backend/tests/pipeline/test_spiders.py`:

```python
def test_match_detail_spider_import():
    from cs2_predictor.pipeline.scraper.spiders.match_detail import MatchDetailSpider
    spider = MatchDetailSpider()
    assert spider.name == "match_detail"
```

- [ ] **Step 2: Rodar teste**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py::test_match_detail_spider_import -v`
Expected: FAIL

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/scraper/spiders/match_detail.py`**

```python
import re
from urllib.parse import urlparse

import scrapy
from scrapy.http import Response

from cs2_predictor.pipeline.scraper.items import MatchDetailItem


class MatchDetailSpider(scrapy.Spider):
    name = "match_detail"
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
                cells = row.css("td::text").getall()
                name = row.css(".st-player a::text").get()
                if not name:
                    continue
                kills_text = row.css(".st-kills::text").get()
                deaths_text = row.css(".st-deaths::text").get()
                adr_text = row.css(".st-adr::text").get()
                kast_text = row.css(".st-kast::text").get()
                rating_text = row.css(".st-rating::text").get()
                players.append({
                    "name": name.strip(),
                    "kills": int(kills_text) if kills_text else None,
                    "deaths": int(deaths_text) if deaths_text else None,
                    "adr": float(adr_text) if adr_text else None,
                    "kast": float(kast_text.rstrip("%")) if kast_text else None,
                    "rating": float(rating_text) if rating_text else None,
                })
            if team_name:
                team_stats.append({"team_name": team_name.strip(), "players": players})

        yield MatchDetailItem(
            hltv_id=match_id,
            team_stats=team_stats,
        )
```

- [ ] **Step 4: Rodar testes**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/scraper/spiders/match_detail.py backend/tests/pipeline/test_spiders.py
git commit -m "feat(scraper): add match detail spider with player stats"
```

---

### Task 6: Runner para executar spiders sob demanda

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/scraper/runner.py`
- Modify: `backend/tests/pipeline/test_spiders.py`

- [ ] **Step 1: Escrever teste**

Adicionar em `backend/tests/pipeline/test_spiders.py`:

```python
def test_runner_import():
    from cs2_predictor.pipeline.scraper.runner import run_scrapers
    assert callable(run_scrapers)
```

- [ ] **Step 2: Rodar teste**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py::test_runner_import -v`
Expected: FAIL

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/scraper/runner.py`**

```python
import logging

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from cs2_predictor.pipeline.scraper.spiders.teams import TeamRankingSpider
from cs2_predictor.pipeline.scraper.spiders.matches import UpcomingMatchesSpider
from cs2_predictor.pipeline.scraper.spiders.results import ResultsSpider
from cs2_predictor.pipeline.scraper.spiders.match_detail import MatchDetailSpider

logger = logging.getLogger(__name__)

SPIDERS = {
    "teams": TeamRankingSpider,
    "matches": UpcomingMatchesSpider,
    "results": ResultsSpider,
    "match_detail": MatchDetailSpider,
}


def run_scrapers(spider_names: list[str], match_ids: list[int] | None = None) -> dict:
    settings = get_project_settings()
    settings.set("ROBOTSTXT_OBEY", False)
    settings.set("CONCURRENT_REQUESTS", 4)
    settings.set("DOWNLOAD_DELAY", 0.5)
    settings.set("COOKIES_ENABLED", False)
    settings.set("ITEM_PIPELINES", {})

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
```

- [ ] **Step 4: Rodar testes**

Run: `cd backend && uv run pytest tests/pipeline/test_spiders.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/scraper/runner.py backend/tests/pipeline/test_spiders.py
git commit -m "feat(scraper): add runner to execute spiders on demand"
```

---

### Task 7: Atualizar pipeline principal (run.py) para usar Scrapy

**Files:**
- Modify: `backend/src/cs2_predictor/pipeline/run.py`
- Remove: `backend/src/cs2_predictor/pipeline/scraper/persistence.py` (não será mais usado diretamente)
- Remove: `backend/tests/pipeline/test_scraper_persistence.py`
- Remove: `backend/tests/pipeline/test_run.py` (substituir)

- [ ] **Step 1: Remover arquivos antigos**

```bash
rm backend/src/cs2_predictor/pipeline/scraper/persistence.py
rm backend/tests/pipeline/test_run.py
```

- [ ] **Step 2: Reescrever `backend/src/cs2_predictor/pipeline/run.py`**

```python
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from cs2_predictor.config import get_settings
from cs2_predictor.db.models import Match, MatchStatus, ModelRun
from cs2_predictor.db.session import get_session_factory
from cs2_predictor.pipeline.features.compute import compute_features_for_scheduled_matches
from cs2_predictor.pipeline.model.calibration import calibrate_platt
from cs2_predictor.pipeline.model.dataset import build_training_dataset
from cs2_predictor.pipeline.model.predict import generate_predictions
from cs2_predictor.pipeline.model.train import train_logistic_regression
from cs2_predictor.pipeline.scraper.runner import run_scrapers

logger = logging.getLogger(__name__)


def _next_version() -> str:
    return f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def run_pipeline(session: Session | None = None, min_matches_to_retrain: int | None = None) -> dict:
    settings = get_settings()
    threshold = min_matches_to_retrain or settings.min_matches_to_retrain
    owned_session = session is None
    if owned_session:
        SessionLocal = get_session_factory()
        session = SessionLocal()
    errors: dict = {}
    retrained = False

    try:
        try:
            result = run_scrapers(["teams", "matches", "results"])
            session.commit()
        except Exception as e:
            logger.exception("scraper failure")
            errors["scraper_error"] = str(e)
            session.rollback()

        try:
            compute_features_for_scheduled_matches(session)
            session.commit()
        except Exception as e:
            logger.exception("feature computation failure")
            errors["features_error"] = str(e)
            session.rollback()

        finished_count = session.query(Match).filter(Match.status == MatchStatus.FINISHED).count()
        if finished_count >= threshold:
            try:
                X, y, feature_names = build_training_dataset(session)
                if len(X) >= threshold and len(set(y.tolist())) > 1:
                    model = train_logistic_regression(X, y, feature_names=feature_names)
                    calibrated = calibrate_platt(model, X, y)
                    version = _next_version()
                    session.add(ModelRun(
                        version=version,
                        trained_at=datetime.now(timezone.utc),
                        accuracy=model.accuracy,
                        features_used=feature_names,
                    ))
                    generate_predictions(session, calibrated, version=version)
                    session.commit()
                    retrained = True
            except Exception as e:
                logger.exception("training failure")
                errors["training_error"] = str(e)
                session.rollback()

        status = "ok" if not errors else "partial"
        return {"status": status, "errors": errors, "retrained": retrained}
    finally:
        if owned_session:
            session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline()
    print(result)
```

- [ ] **Step 3: Commit**

```bash
git rm backend/src/cs2_predictor/pipeline/scraper/persistence.py
git rm backend/tests/pipeline/test_run.py
git add backend/src/cs2_predictor/pipeline/run.py
git commit -m "feat(pipeline): integrate Scrapy runner, remove old scraper"
```

---

### Task 8: Atualizar features/compute para usar dados do Scrapy

**Files:**
- Modify: `backend/src/cs2_predictor/pipeline/features/compute.py`

- [ ] **Step 1: Atualizar imports em `compute.py`**

Substituir `from cs2_predictor.pipeline.scraper.hltv import HLTVScraper` por import do runner.

- [ ] **Step 2: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/features/compute.py
git commit -m "refactor: update compute.py imports for new scraper structure"
```

---

### Task 9: Limpeza e verificação final

- [ ] **Step 1: Rodar suite completa de testes**

```bash
cd backend && uv run pytest -v
```

Expected: todos os testes PASSED (ajustar conforme necessário).

- [ ] **Step 2: Rodar pipeline end-to-end**

```bash
cd backend && uv run python -m cs2_predictor.pipeline.run
```

Expected: pipeline executa sem erros.

- [ ] **Step 3: Commit final**

```bash
git tag v0.2.0
```

- [ ] **Step 4: Push**

```bash
git push origin main --force --tags
```
