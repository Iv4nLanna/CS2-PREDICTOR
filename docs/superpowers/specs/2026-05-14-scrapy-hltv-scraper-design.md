# Scrapy HLTV Scraper — Design Spec

## Objetivo

Substituir o scraper atual (baseado em `eupeutro/hltv-api`) por um scraper próprio em **Scrapy** que extrai dados diretamente do HLTV.org, incluindo stats detalhados por jogador (Rating, K/D, ADR, KAST).

## Arquitetura

```
backend/src/cs2_predictor/pipeline/scraper/
├── __init__.py
├── persistence.py          # Mantido — upsert no banco
├── items.py                # NOVO — definição dos dados extraídos
├── pipelines.py            # NOVO — persistência via SQLAlchemy
├── runner.py               # NOVO — executa spiders sob demanda
├── spiders/
│   ├── __init__.py
│   ├── teams.py            # Ranking HLTV + perfil dos times
│   ├── matches.py          # Partidas agendadas
│   ├── results.py          # Resultados históricos (paginado)
│   └── match_detail.py     # Stats por jogador em partidas finalizadas
└── middleware.py            # NOVO — rate limiting + headers
```

## Spiders

### 1. Teams Spider

**Páginas:**
- `https://www.hltv.org/ranking/team/` — ranking mundial
- `https://www.hltv.org/team/{id}/{slug}` — perfil do time

**Dados extraídos:**

```python
class TeamItem(Item):
    hltv_id = IntegerField()
    name = StringField()
    country = StringField()
    rank = IntegerField()
    players = ListField()  # [{"name": str, "rating": float, "role": str}]
```

### 2. Matches Spider

**Página:** `https://www.hltv.org/matches`

**Dados extraídos:**

```python
class MatchItem(Item):
    hltv_id = IntegerField()
    team_a_id = IntegerField()
    team_b_id = IntegerField()
    team_a_name = StringField()
    team_b_name = StringField()
    format = StringField()           # BO1, BO3, BO5
    is_lan = BooleanField()
    event_name = StringField()
    scheduled_at = DateTimeField()
    map_pool = ListField()
```

### 3. Results Spider

**Página:** `https://www.hltv.org/results` (com paginação via `?offset=100`)

**Dados extraídos:**

```python
class ResultItem(Item):
    hltv_id = IntegerField()
    team_a_id = IntegerField()
    team_b_id = IntegerField()
    team_a_name = StringField()
    team_b_name = StringField()
    team_a_score = IntegerField()
    team_b_score = IntegerField()
    format = StringField()
    event_name = StringField()
    played_at = DateTimeField()
    maps = ListField()              # [{"name": str, "score_a": int, "score_b": int}]
```

### 4. Match Detail Spider

**Página:** `https://www.hltv.org/matches/{id}/{slug}`

**Dados extraídos (por partida finalizada):**

```python
class MatchDetailItem(Item):
    hltv_id = IntegerField()
    team_stats = ListField()        # [
                                    #   {"team_name": str,
                                    #    "players": [
                                    #       {"name": str, "kills": int, "deaths": int,
                                    #        "adr": float, "kast": float, "rating": float}
                                    #    ]}
                                    # ]
    map_results = ListField()       # [{"map_name": str, "team_a_score": int, "team_b_score": int}]
```

## Persistência

O `pipelines.py` contém pipelines que recebem os `Item` extraídos e chamam as funções em `persistence.py` para salvar no banco.

A pipeline de MatchDetail salva stats dos jogadores em uma nova tabela:

```sql
CREATE TABLE player_match_stats (
    id SERIAL PRIMARY KEY,
    match_id INTEGER REFERENCES matches(id),
    team_id INTEGER REFERENCES teams(id),
    player_name VARCHAR(200),
    kills INTEGER,
    deaths INTEGER,
    adr FLOAT,
    kast FLOAT,
    rating FLOAT,
    UNIQUE(match_id, team_id, player_name)
);
```

## Execução

O `runner.py` expõe:

```python
def run_scrapers(spiders: list[str], session: Session) -> dict:
    """Executa spiders especificados. spiders=['teams','matches','results','match_detail']"""
```

O pipeline (`run.py`) chama o runner e depois features/modelo:

```python
def run_pipeline(session=None):
    scraper_result = run_scrapers(['teams', 'matches', 'results', 'match_detail'], session)
    compute_features_for_scheduled_matches(session)
    # ... treino e predição ...
```

## Dependências

Adicionar ao `pyproject.toml`:
- `scrapy>=2.11`
- Será usado via `scrapy crawl` ou API da Scrapy (`CrawlerProcess`)

## Limpeza

Após implementação, remover:
- `hltv.py` (scraper antigo baseado em eupeutro API)
- `tests/pipeline/test_hltv_scraper.py` (testes do scraper antigo)
- `tests/pipeline/test_scraper_persistence.py` (testes de persistência serão ajustados)

## Critérios de Sucesso

1. Scrapy extrai ranking de times, partidas, resultados e match-details do HLTV
2. Dados são persistidos no banco via pipelines
3. Match detail extrai stats de jogadores (rating, K/D, ADR, KAST)
4. Pipeline (`run.py`) funciona end-to-end com o novo scraper
5. Todos os testes passam (incluindo novos testes do scraper)
