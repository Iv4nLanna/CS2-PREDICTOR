# CS2 Win Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Before implementing any feature/bugfix, invoke `superpowers:test-driven-development`. For any failing test or unexpected error, invoke `superpowers:systematic-debugging`. Before claiming a task complete, invoke `superpowers:verification-before-completion`.

**Goal:** Construir plataforma web que prevê probabilidade de vitória em partidas profissionais de CS2 usando regressão logística calibrada sobre dados do HLTV.

**Architecture:** Pipeline agendado (GitHub Actions) coleta dados do HLTV, computa features e grava previsões no Postgres. API FastAPI read-only serve as previsões. Frontend Next.js consome a API e exibe dashboard.

**Tech Stack:** Python 3.12 (uv), SQLAlchemy 2.x, Alembic, FastAPI, scikit-learn, pandas, pytest, Next.js 15 (App Router, TypeScript, Tailwind), pnpm, GitHub Actions, Neon (Postgres), Render, Vercel.

**Spec reference:** `docs/superpowers/specs/2026-05-13-cs2-predictor-design.md`

**Project structure:**
```
cs2-predictor/
├── backend/                          # Pipeline + API (Python, shared DB models)
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── alembic/versions/
│   ├── src/cs2_predictor/
│   │   ├── config.py
│   │   ├── db/                       # SQLAlchemy models + session
│   │   ├── pipeline/                 # scraper, features, model, run.py
│   │   └── api/                      # FastAPI app
│   └── tests/
├── frontend/                         # Next.js 15
│   ├── package.json
│   ├── app/
│   ├── components/
│   └── lib/
└── .github/workflows/pipeline.yml    # cron a cada 6h
```

**Pré-requisitos do executor:**
- Python 3.12+ instalado
- `uv` instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `pnpm` instalado (`npm i -g pnpm`)
- Postgres local rodando OU connection string Neon — exportar `DATABASE_URL` e `DATABASE_URL_TEST`
- Working directory: `/home/ivan/cs2-predictor`

---

## Phase 1: Foundation

### Task 1: Inicializar projeto backend

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/cs2_predictor/__init__.py`
- Create: `backend/src/cs2_predictor/config.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/.env.example`
- Create: `.gitignore`

- [ ] **Step 1: Criar `.gitignore` na raiz**

```
# Python
__pycache__/
*.py[cod]
.venv/
.env
*.egg-info/
.pytest_cache/
.coverage
htmlcov/

# Node
node_modules/
.next/
out/

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

- [ ] **Step 2: Criar `backend/pyproject.toml`**

```toml
[project]
name = "cs2-predictor"
version = "0.1.0"
description = "CS2 professional match win predictor"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "psycopg[binary]>=3.2",
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "httpx>=0.27",
    "pandas>=2.2",
    "scikit-learn>=1.5",
    "numpy>=2.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/cs2_predictor"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

- [ ] **Step 3: Criar arquivos vazios de package**

```bash
mkdir -p backend/src/cs2_predictor backend/tests backend/alembic/versions
touch backend/src/cs2_predictor/__init__.py backend/tests/__init__.py
```

- [ ] **Step 4: Criar `backend/src/cs2_predictor/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    database_url_test: str = ""
    hltv_api_base_url: str = "http://localhost:8000"
    pipeline_interval_hours: int = 6
    min_matches_to_retrain: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Criar `backend/.env.example`**

```
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/cs2_predictor
DATABASE_URL_TEST=postgresql+psycopg://user:pass@localhost:5432/cs2_predictor_test
HLTV_API_BASE_URL=http://localhost:8000
PIPELINE_INTERVAL_HOURS=6
MIN_MATCHES_TO_RETRAIN=10
```

- [ ] **Step 6: Criar `backend/tests/conftest.py`**

```python
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture(scope="session")
def test_engine():
    url = os.environ.get("DATABASE_URL_TEST")
    if not url:
        pytest.skip("DATABASE_URL_TEST not set")
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine) -> Session:
    from cs2_predictor.db.models import Base

    Base.metadata.drop_all(test_engine)
    Base.metadata.create_all(test_engine)
    SessionLocal = sessionmaker(bind=test_engine, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(test_engine)
```

- [ ] **Step 7: Instalar dependências**

Run: `cd backend && uv venv && uv pip install -e ".[dev]"`
Expected: env criado em `.venv`, deps instaladas sem erro.

- [ ] **Step 8: Commit**

```bash
git add .gitignore backend/
git commit -m "feat: bootstrap backend project with uv and pyproject"
```

---

### Task 2: Definir modelos SQLAlchemy

**Files:**
- Create: `backend/src/cs2_predictor/db/__init__.py`
- Create: `backend/src/cs2_predictor/db/models.py`
- Create: `backend/src/cs2_predictor/db/session.py`
- Create: `backend/tests/db/__init__.py`
- Create: `backend/tests/db/test_models.py`

Invoke `superpowers:test-driven-development` antes de implementar.

- [ ] **Step 1: Criar `backend/src/cs2_predictor/db/__init__.py` (vazio)**

```bash
mkdir -p backend/src/cs2_predictor/db backend/tests/db
touch backend/src/cs2_predictor/db/__init__.py backend/tests/db/__init__.py
```

- [ ] **Step 2: Escrever teste que falha — `backend/tests/db/test_models.py`**

```python
from datetime import datetime, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    ModelRun,
    Player,
    Prediction,
    Team,
    TeamFeatures,
)


def test_team_persists(db_session):
    team = Team(hltv_id=4608, name="Natus Vincere", country="Ukraine", hltv_ranking=1)
    db_session.add(team)
    db_session.commit()
    fetched = db_session.query(Team).filter_by(hltv_id=4608).one()
    assert fetched.name == "Natus Vincere"


def test_match_links_two_teams(db_session):
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100,
        team_a_id=a.id,
        team_b_id=b.id,
        format=MatchFormat.BO3,
        is_lan=True,
        map_pool=["de_mirage", "de_inferno", "de_anubis"],
        tournament="Major",
        scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.commit()
    fetched = db_session.query(Match).filter_by(hltv_id=100).one()
    assert fetched.team_a.name == "A"
    assert fetched.team_b.name == "B"
    assert fetched.map_pool == ["de_mirage", "de_inferno", "de_anubis"]


def test_prediction_links_to_match_and_model_run(db_session):
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.SCHEDULED,
    )
    run = ModelRun(version="v1", trained_at=datetime.now(timezone.utc),
                   accuracy=0.65, features_used=["win_rate_recent_decayed"])
    db_session.add_all([match, run])
    db_session.flush()
    pred = Prediction(match_id=match.id, team_a_win_prob=0.6,
                      team_b_win_prob=0.4, model_version="v1", calibrated=True)
    db_session.add(pred)
    db_session.commit()
    fetched = db_session.query(Prediction).one()
    assert fetched.team_a_win_prob == 0.6
    assert fetched.match.hltv_id == 100


def test_player_optional_faceit_columns(db_session):
    team = Team(hltv_id=1, name="A")
    db_session.add(team)
    db_session.flush()
    player = Player(team_id=team.id, hltv_id=999, name="s1mple",
                     rating=1.20, role="rifler", active=True)
    db_session.add(player)
    db_session.commit()
    fetched = db_session.query(Player).filter_by(hltv_id=999).one()
    assert fetched.faceit_id is None
    assert fetched.faceit_rating is None


def test_team_features_per_match(db_session):
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.flush()
    tf = TeamFeatures(
        team_id=a.id, match_id=match.id,
        win_rate_recent_decayed=0.7,
        head_to_head_decayed=0.5,
        hltv_ranking_snapshot=1,
        sos_score=0.8,
        map_stats={"de_mirage": 0.75},
    )
    db_session.add(tf)
    db_session.commit()
    fetched = db_session.query(TeamFeatures).one()
    assert fetched.map_stats["de_mirage"] == 0.75


def test_match_result_unique_per_match(db_session):
    from sqlalchemy.exc import IntegrityError

    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=datetime.now(timezone.utc),
        status=MatchStatus.FINISHED,
    )
    db_session.add(match)
    db_session.flush()
    r1 = MatchResult(match_id=match.id, winner_id=a.id,
                     score_detail={"de_mirage": [16, 12]},
                     played_at=datetime.now(timezone.utc))
    db_session.add(r1)
    db_session.commit()
    r2 = MatchResult(match_id=match.id, winner_id=b.id,
                     score_detail={}, played_at=datetime.now(timezone.utc))
    db_session.add(r2)
    try:
        db_session.commit()
        raise AssertionError("expected IntegrityError")
    except IntegrityError:
        db_session.rollback()
```

- [ ] **Step 3: Rodar teste — deve falhar com ImportError**

Run: `cd backend && uv run pytest tests/db/test_models.py -v`
Expected: FAIL — `ImportError` ou `ModuleNotFoundError` (módulo `db.models` ainda não existe).

- [ ] **Step 4: Implementar `backend/src/cs2_predictor/db/models.py`**

```python
import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MatchFormat(str, enum.Enum):
    BO1 = "BO1"
    BO3 = "BO3"
    BO5 = "BO5"


class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    hltv_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)
    hltv_ranking: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    hltv_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    faceit_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    faceit_rating: Mapped[float | None] = mapped_column(Float, nullable=True)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    hltv_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    team_a_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    team_b_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    format: Mapped[MatchFormat] = mapped_column(Enum(MatchFormat))
    is_lan: Mapped[bool] = mapped_column(Boolean, default=False)
    map_pool: Mapped[list] = mapped_column(JSON, default=list)
    tournament: Mapped[str | None] = mapped_column(String(200), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[MatchStatus] = mapped_column(Enum(MatchStatus), default=MatchStatus.SCHEDULED)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    team_a = relationship("Team", foreign_keys=[team_a_id])
    team_b = relationship("Team", foreign_keys=[team_b_id])


class MatchResult(Base):
    __tablename__ = "match_results"
    __table_args__ = (UniqueConstraint("match_id", name="uq_match_result_match"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    winner_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    score_detail: Mapped[dict] = mapped_column(JSON, default=dict)
    played_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TeamFeatures(Base):
    __tablename__ = "team_features"
    __table_args__ = (UniqueConstraint("team_id", "match_id", name="uq_team_features_team_match"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    win_rate_recent_decayed: Mapped[float] = mapped_column(Float)
    head_to_head_decayed: Mapped[float] = mapped_column(Float)
    hltv_ranking_snapshot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sos_score: Mapped[float] = mapped_column(Float)
    map_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accuracy: Mapped[float] = mapped_column(Float)
    features_used: Mapped[list] = mapped_column(JSON, default=list)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    team_a_win_prob: Mapped[float] = mapped_column(Float)
    team_b_win_prob: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
    calibrated: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    match = relationship("Match")
```

- [ ] **Step 5: Criar `backend/src/cs2_predictor/db/session.py`**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cs2_predictor.config import get_settings


def get_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db() -> Session:
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 6: Rodar teste — deve passar**

Run: `cd backend && uv run pytest tests/db/test_models.py -v`
Expected: 6 PASSED.

- [ ] **Step 7: Commit**

```bash
git add backend/src/cs2_predictor/db/ backend/tests/db/
git commit -m "feat(db): add SQLAlchemy models for teams, matches, features, predictions"
```

---

### Task 3: Configurar Alembic e gerar migration inicial

**Files:**
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/0001_initial.py` (gerado)

- [ ] **Step 1: Inicializar Alembic**

Run: `cd backend && uv run alembic init alembic`
Expected: Alembic cria `alembic.ini` e diretório `alembic/` com `env.py`, `script.py.mako`.

- [ ] **Step 2: Editar `backend/alembic.ini`** — substituir `sqlalchemy.url` por placeholder vazio

Localizar a linha `sqlalchemy.url = driver://user:pass@localhost/dbname` e trocar por:

```ini
sqlalchemy.url =
```

(será injetada via `env.py`).

- [ ] **Step 3: Substituir `backend/alembic/env.py` por esta versão**

```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from cs2_predictor.config import get_settings
from cs2_predictor.db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Gerar migration inicial**

Run: `cd backend && uv run alembic revision --autogenerate -m "initial schema"`
Expected: novo arquivo em `alembic/versions/` (renomear para `0001_initial.py` se desejar).

- [ ] **Step 5: Aplicar migration**

Run: `cd backend && uv run alembic upgrade head`
Expected: tabelas criadas no `DATABASE_URL`. Verificar com `psql $DATABASE_URL -c "\dt"`.

- [ ] **Step 6: Commit**

```bash
git add backend/alembic.ini backend/alembic/
git commit -m "feat(db): add alembic with initial schema migration"
```

---

## Phase 2: Pipeline — Scraper

### Task 4: Cliente HTTP para `eupeutro/hltv-api`

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/__init__.py`
- Create: `backend/src/cs2_predictor/pipeline/scraper/__init__.py`
- Create: `backend/src/cs2_predictor/pipeline/scraper/hltv.py`
- Create: `backend/tests/pipeline/__init__.py`
- Create: `backend/tests/pipeline/test_hltv_scraper.py`

Invoke `superpowers:test-driven-development` antes de implementar.

- [ ] **Step 1: Criar diretórios**

```bash
mkdir -p backend/src/cs2_predictor/pipeline/scraper backend/tests/pipeline
touch backend/src/cs2_predictor/pipeline/__init__.py \
      backend/src/cs2_predictor/pipeline/scraper/__init__.py \
      backend/tests/pipeline/__init__.py
```

- [ ] **Step 2: Escrever teste — `backend/tests/pipeline/test_hltv_scraper.py`**

```python
from unittest.mock import patch

import pytest

from cs2_predictor.pipeline.scraper.hltv import HLTVScraper, ScraperError


def _mock_response(json_data, status_code=200):
    class R:
        def __init__(self, data, code):
            self._data = data
            self.status_code = code

        def json(self):
            return self._data

        def raise_for_status(self):
            if self.status_code >= 400:
                import httpx
                raise httpx.HTTPStatusError("err", request=None, response=self)

    return R(json_data, status_code)


def test_fetch_team_ranking_returns_list():
    scraper = HLTVScraper(base_url="http://fake")
    payload = [{"id": 1, "name": "Navi", "rank": 1, "country": "UA"}]
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        result = scraper.fetch_team_ranking()
    assert result == payload


def test_fetch_team_ranking_raises_on_http_error():
    scraper = HLTVScraper(base_url="http://fake")
    with patch("httpx.Client.get", return_value=_mock_response({}, status_code=500)):
        with pytest.raises(ScraperError):
            scraper.fetch_team_ranking()


def test_fetch_upcoming_matches():
    scraper = HLTVScraper(base_url="http://fake")
    payload = [{"id": 100, "team_a_id": 1, "team_b_id": 2, "format": "BO3"}]
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        result = scraper.fetch_upcoming_matches()
    assert result[0]["id"] == 100


def test_fetch_match_results_since():
    scraper = HLTVScraper(base_url="http://fake")
    payload = [{"id": 100, "winner_id": 1}]
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        result = scraper.fetch_match_results(since_days=7)
    assert len(result) == 1


def test_fetch_team_detail():
    scraper = HLTVScraper(base_url="http://fake")
    payload = {"id": 1, "name": "Navi", "players": []}
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        result = scraper.fetch_team(team_id=1)
    assert result["name"] == "Navi"
```

- [ ] **Step 3: Rodar — deve falhar com ImportError**

Run: `cd backend && uv run pytest tests/pipeline/test_hltv_scraper.py -v`
Expected: FAIL (módulo ainda não existe).

- [ ] **Step 4: Implementar `backend/src/cs2_predictor/pipeline/scraper/hltv.py`**

```python
import httpx


class ScraperError(Exception):
    pass


class HLTVScraper:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, base_url=self.base_url)

    def _get(self, path: str, params: dict | None = None):
        try:
            response = self._client.get(path, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ScraperError(f"HLTV API request failed: {path}: {e}") from e

    def fetch_team_ranking(self) -> list[dict]:
        return self._get("/ranking/teams")

    def fetch_upcoming_matches(self) -> list[dict]:
        return self._get("/matches/upcoming")

    def fetch_match_results(self, since_days: int = 7) -> list[dict]:
        return self._get("/matches/results", params={"since_days": since_days})

    def fetch_team(self, team_id: int) -> dict:
        return self._get(f"/teams/{team_id}")

    def close(self):
        self._client.close()
```

- [ ] **Step 5: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/test_hltv_scraper.py -v`
Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/ backend/tests/pipeline/
git commit -m "feat(pipeline): add HLTV API HTTP client"
```

---

### Task 5: Persistência do scraper no banco

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/scraper/persistence.py`
- Create: `backend/tests/pipeline/test_scraper_persistence.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/test_scraper_persistence.py`**

```python
from datetime import datetime, timezone

from cs2_predictor.db.models import Match, MatchFormat, MatchResult, MatchStatus, Team
from cs2_predictor.pipeline.scraper.persistence import (
    upsert_match_results,
    upsert_matches,
    upsert_teams,
)


def test_upsert_teams_inserts_new(db_session):
    payload = [
        {"id": 1, "name": "Navi", "country": "UA", "rank": 1},
        {"id": 2, "name": "FaZe", "country": "EU", "rank": 2},
    ]
    upsert_teams(db_session, payload)
    db_session.commit()
    assert db_session.query(Team).count() == 2


def test_upsert_teams_updates_existing(db_session):
    db_session.add(Team(hltv_id=1, name="Old", country="UA", hltv_ranking=5))
    db_session.commit()
    upsert_teams(db_session, [{"id": 1, "name": "Navi", "country": "UA", "rank": 1}])
    db_session.commit()
    team = db_session.query(Team).filter_by(hltv_id=1).one()
    assert team.name == "Navi"
    assert team.hltv_ranking == 1


def test_upsert_matches_creates_with_teams(db_session):
    db_session.add_all([Team(hltv_id=1, name="A"), Team(hltv_id=2, name="B")])
    db_session.commit()
    payload = [{
        "id": 100,
        "team_a_id": 1,
        "team_b_id": 2,
        "format": "BO3",
        "is_lan": True,
        "map_pool": ["de_mirage"],
        "tournament": "Major",
        "scheduled_at": "2026-06-01T18:00:00+00:00",
    }]
    upsert_matches(db_session, payload)
    db_session.commit()
    match = db_session.query(Match).filter_by(hltv_id=100).one()
    assert match.format == MatchFormat.BO3
    assert match.is_lan is True


def test_upsert_match_results_marks_finished(db_session):
    db_session.add_all([Team(hltv_id=1, name="A"), Team(hltv_id=2, name="B")])
    db_session.commit()
    upsert_matches(db_session, [{
        "id": 100, "team_a_id": 1, "team_b_id": 2, "format": "BO1",
        "is_lan": False, "map_pool": [], "tournament": "x",
        "scheduled_at": "2026-06-01T18:00:00+00:00",
    }])
    db_session.commit()
    upsert_match_results(db_session, [{
        "id": 100,
        "winner_id": 1,
        "score_detail": {"de_mirage": [16, 12]},
        "played_at": "2026-06-01T20:00:00+00:00",
    }])
    db_session.commit()
    match = db_session.query(Match).filter_by(hltv_id=100).one()
    assert match.status == MatchStatus.FINISHED
    result = db_session.query(MatchResult).filter_by(match_id=match.id).one()
    assert result.score_detail == {"de_mirage": [16, 12]}
```

- [ ] **Step 2: Rodar — falha por import**

Run: `cd backend && uv run pytest tests/pipeline/test_scraper_persistence.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/scraper/persistence.py`**

```python
from datetime import datetime

from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def upsert_teams(session: Session, payload: list[dict]) -> None:
    for item in payload:
        team = session.query(Team).filter_by(hltv_id=item["id"]).one_or_none()
        if team is None:
            team = Team(hltv_id=item["id"])
            session.add(team)
        team.name = item.get("name", team.name if team.id else "")
        team.country = item.get("country")
        team.hltv_ranking = item.get("rank")
        team.updated_at = datetime.utcnow()


def upsert_matches(session: Session, payload: list[dict]) -> None:
    for item in payload:
        team_a = session.query(Team).filter_by(hltv_id=item["team_a_id"]).one()
        team_b = session.query(Team).filter_by(hltv_id=item["team_b_id"]).one()
        match = session.query(Match).filter_by(hltv_id=item["id"]).one_or_none()
        if match is None:
            match = Match(hltv_id=item["id"])
            session.add(match)
        match.team_a_id = team_a.id
        match.team_b_id = team_b.id
        match.format = MatchFormat(item["format"])
        match.is_lan = bool(item.get("is_lan", False))
        match.map_pool = item.get("map_pool", [])
        match.tournament = item.get("tournament")
        match.scheduled_at = _parse_dt(item["scheduled_at"])
        if match.status is None:
            match.status = MatchStatus.SCHEDULED


def upsert_match_results(session: Session, payload: list[dict]) -> None:
    for item in payload:
        match = session.query(Match).filter_by(hltv_id=item["id"]).one()
        winner = session.query(Team).filter_by(hltv_id=item["winner_id"]).one()
        result = session.query(MatchResult).filter_by(match_id=match.id).one_or_none()
        if result is None:
            result = MatchResult(match_id=match.id)
            session.add(result)
        result.winner_id = winner.id
        result.score_detail = item.get("score_detail", {})
        result.played_at = _parse_dt(item["played_at"])
        match.status = MatchStatus.FINISHED
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/test_scraper_persistence.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/scraper/persistence.py backend/tests/pipeline/test_scraper_persistence.py
git commit -m "feat(pipeline): upsert teams, matches and results into Postgres"
```

---

## Phase 2: Pipeline — Feature Engineering

### Task 6: Feature `win_rate_recent_decayed`

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/features/__init__.py`
- Create: `backend/src/cs2_predictor/pipeline/features/recent_form.py`
- Create: `backend/tests/pipeline/features/__init__.py`
- Create: `backend/tests/pipeline/features/test_recent_form.py`

- [ ] **Step 1: Criar diretórios**

```bash
mkdir -p backend/src/cs2_predictor/pipeline/features backend/tests/pipeline/features
touch backend/src/cs2_predictor/pipeline/features/__init__.py \
      backend/tests/pipeline/features/__init__.py
```

- [ ] **Step 2: Escrever teste — `backend/tests/pipeline/features/test_recent_form.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.pipeline.features.recent_form import compute_recent_form


def _match(date, won):
    return {"played_at": date, "won": won}


def test_no_matches_returns_neutral():
    assert compute_recent_form([], reference_date=datetime.now(timezone.utc)) == 0.5


def test_all_wins_recent_returns_close_to_one():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [_match(ref - timedelta(days=i), won=True) for i in range(1, 6)]
    score = compute_recent_form(matches, reference_date=ref, half_life_days=30)
    assert score > 0.9


def test_all_losses_returns_close_to_zero():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [_match(ref - timedelta(days=i), won=False) for i in range(1, 6)]
    score = compute_recent_form(matches, reference_date=ref, half_life_days=30)
    assert score < 0.1


def test_recent_match_weighs_more_than_old():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    recent_win = [_match(ref - timedelta(days=1), True)]
    old_win = [_match(ref - timedelta(days=180), True)]
    recent_score = compute_recent_form(recent_win, reference_date=ref, half_life_days=30)
    old_score = compute_recent_form(old_win, reference_date=ref, half_life_days=30)
    assert recent_score > old_score


def test_mix_returns_between_zero_and_one():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [
        _match(ref - timedelta(days=1), True),
        _match(ref - timedelta(days=2), False),
        _match(ref - timedelta(days=3), True),
    ]
    score = compute_recent_form(matches, reference_date=ref)
    assert 0.0 < score < 1.0
```

- [ ] **Step 3: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/features/test_recent_form.py -v`
Expected: FAIL — ImportError.

- [ ] **Step 4: Implementar `backend/src/cs2_predictor/pipeline/features/recent_form.py`**

```python
import math
from datetime import datetime


def compute_recent_form(
    matches: list[dict],
    reference_date: datetime,
    half_life_days: float = 30.0,
) -> float:
    """Weighted win rate with exponential decay.

    Each match contributes weight = 0.5 ** (age_days / half_life_days).
    Returns 0.5 (neutral) if no matches.
    """
    if not matches:
        return 0.5

    weighted_wins = 0.0
    total_weight = 0.0
    for m in matches:
        age_days = (reference_date - m["played_at"]).total_seconds() / 86400.0
        if age_days < 0:
            continue
        weight = math.pow(0.5, age_days / half_life_days)
        total_weight += weight
        if m["won"]:
            weighted_wins += weight

    if total_weight == 0:
        return 0.5
    return weighted_wins / total_weight
```

- [ ] **Step 5: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/features/test_recent_form.py -v`
Expected: 5 PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/features/recent_form.py backend/tests/pipeline/features/
git commit -m "feat(features): add exponentially-decayed recent form"
```

---

### Task 7: Feature `win_rate_per_map` (map_stats)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/features/map_stats.py`
- Create: `backend/tests/pipeline/features/test_map_stats.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/features/test_map_stats.py`**

```python
from cs2_predictor.pipeline.features.map_stats import compute_map_stats


def test_empty_returns_empty_dict():
    assert compute_map_stats([], map_pool=["de_mirage"]) == {"de_mirage": 0.5}


def test_single_map_winrate():
    matches = [
        {"map": "de_mirage", "won": True},
        {"map": "de_mirage", "won": True},
        {"map": "de_mirage", "won": False},
    ]
    result = compute_map_stats(matches, map_pool=["de_mirage"])
    assert abs(result["de_mirage"] - 2 / 3) < 1e-9


def test_only_maps_in_pool_returned():
    matches = [
        {"map": "de_mirage", "won": True},
        {"map": "de_inferno", "won": False},
    ]
    result = compute_map_stats(matches, map_pool=["de_mirage"])
    assert "de_inferno" not in result
    assert "de_mirage" in result


def test_pool_map_with_no_history_returns_neutral():
    result = compute_map_stats([], map_pool=["de_anubis", "de_nuke"])
    assert result == {"de_anubis": 0.5, "de_nuke": 0.5}
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/features/test_map_stats.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/features/map_stats.py`**

```python
from collections import defaultdict


def compute_map_stats(matches: list[dict], map_pool: list[str]) -> dict[str, float]:
    """Win rate per map, restricted to the given map_pool.

    Maps in the pool with no history return 0.5 (neutral).
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [wins, total]
    for m in matches:
        bucket = counts[m["map"]]
        bucket[1] += 1
        if m["won"]:
            bucket[0] += 1

    result: dict[str, float] = {}
    for map_name in map_pool:
        wins, total = counts.get(map_name, (0, 0))
        result[map_name] = wins / total if total > 0 else 0.5
    return result
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/features/test_map_stats.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/features/map_stats.py backend/tests/pipeline/features/test_map_stats.py
git commit -m "feat(features): add per-map win rate restricted to veto pool"
```

---

### Task 8: Feature `head_to_head_decayed`

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/features/head_to_head.py`
- Create: `backend/tests/pipeline/features/test_head_to_head.py`

- [ ] **Step 1: Escrever teste**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.pipeline.features.head_to_head import compute_head_to_head


def test_no_history_returns_neutral():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert compute_head_to_head([], reference_date=ref) == 0.5


def test_perfect_record_recent():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    matches = [
        {"played_at": ref - timedelta(days=10), "team_won": True}
        for _ in range(3)
    ]
    score = compute_head_to_head(matches, reference_date=ref)
    assert score > 0.9


def test_old_match_weighs_less_than_recent():
    ref = datetime(2026, 5, 1, tzinfo=timezone.utc)
    recent = [{"played_at": ref - timedelta(days=10), "team_won": True}]
    old = [{"played_at": ref - timedelta(days=400), "team_won": True}]
    assert compute_head_to_head(recent, reference_date=ref) > compute_head_to_head(old, reference_date=ref)
```

Salvar em `backend/tests/pipeline/features/test_head_to_head.py`.

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/features/test_head_to_head.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/features/head_to_head.py`**

```python
import math
from datetime import datetime


def compute_head_to_head(
    matches: list[dict],
    reference_date: datetime,
    half_life_days: float = 180.0,
) -> float:
    """Win rate against a specific opponent with time decay.

    half_life_days is larger here than recent_form because H2H signal is rare.
    """
    if not matches:
        return 0.5

    weighted_wins = 0.0
    total_weight = 0.0
    for m in matches:
        age_days = (reference_date - m["played_at"]).total_seconds() / 86400.0
        if age_days < 0:
            continue
        weight = math.pow(0.5, age_days / half_life_days)
        total_weight += weight
        if m["team_won"]:
            weighted_wins += weight

    return weighted_wins / total_weight if total_weight > 0 else 0.5
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/features/test_head_to_head.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/features/head_to_head.py backend/tests/pipeline/features/test_head_to_head.py
git commit -m "feat(features): add head-to-head with time decay"
```

---

### Task 9: Feature `sos_score` (Strength of Schedule)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/features/sos.py`
- Create: `backend/tests/pipeline/features/test_sos.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/features/test_sos.py`**

```python
from cs2_predictor.pipeline.features.sos import compute_sos


def test_no_matches_returns_zero():
    assert compute_sos([]) == 0.0


def test_wins_against_strong_opponents_score_high():
    matches = [
        {"opponent_rank": 5, "won": True},
        {"opponent_rank": 8, "won": True},
    ]
    weak = [
        {"opponent_rank": 100, "won": True},
        {"opponent_rank": 80, "won": True},
    ]
    assert compute_sos(matches) > compute_sos(weak)


def test_losses_count_against():
    wins = [{"opponent_rank": 10, "won": True}]
    losses = [{"opponent_rank": 10, "won": False}]
    assert compute_sos(wins) > compute_sos(losses)


def test_unranked_opponent_treated_as_weak():
    ranked = [{"opponent_rank": 5, "won": True}]
    unranked = [{"opponent_rank": None, "won": True}]
    assert compute_sos(ranked) > compute_sos(unranked)
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/features/test_sos.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/features/sos.py`**

```python
UNRANKED_PROXY = 150


def _opponent_weight(rank: int | None) -> float:
    """Higher rank number = weaker opponent. Top-1 = weight 1.0, top-100 ~= 0.1."""
    effective = rank if rank is not None else UNRANKED_PROXY
    effective = max(effective, 1)
    return 1.0 / (1.0 + 0.05 * (effective - 1))


def compute_sos(matches: list[dict]) -> float:
    """Strength of Schedule: opponent-weighted win rate.

    A win against rank 1 contributes far more than a win against rank 100.
    A loss subtracts the same weight a win would add (signed scoring).
    """
    if not matches:
        return 0.0

    total_score = 0.0
    total_weight = 0.0
    for m in matches:
        w = _opponent_weight(m.get("opponent_rank"))
        total_weight += w
        total_score += w if m["won"] else -w

    if total_weight == 0:
        return 0.0
    return (total_score / total_weight + 1.0) / 2.0
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/features/test_sos.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/features/sos.py backend/tests/pipeline/features/test_sos.py
git commit -m "feat(features): add strength-of-schedule score"
```

---

### Task 10: Orquestrador que computa e persiste `TeamFeatures` por partida

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/features/compute.py`
- Create: `backend/tests/pipeline/features/test_compute.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/features/test_compute.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.features.compute import compute_features_for_scheduled_matches


def _seed(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", hltv_ranking=20)
    c = Team(hltv_id=3, name="C", hltv_ranking=50)
    db_session.add_all([a, b, c])
    db_session.flush()
    past = Match(
        hltv_id=10, team_a_id=a.id, team_b_id=c.id, format=MatchFormat.BO1,
        is_lan=False, map_pool=["de_mirage"], tournament="x",
        scheduled_at=now - timedelta(days=20), status=MatchStatus.FINISHED,
    )
    db_session.add(past)
    db_session.flush()
    db_session.add(MatchResult(
        match_id=past.id, winner_id=a.id,
        score_detail={"de_mirage": [16, 10]},
        played_at=now - timedelta(days=20),
    ))
    upcoming = Match(
        hltv_id=11, team_a_id=a.id, team_b_id=b.id, format=MatchFormat.BO3,
        is_lan=True, map_pool=["de_mirage", "de_inferno"], tournament="x",
        scheduled_at=now + timedelta(days=2), status=MatchStatus.SCHEDULED,
    )
    db_session.add(upcoming)
    db_session.commit()
    return a, b, upcoming


def test_creates_two_team_feature_rows_per_match(db_session):
    a, b, upcoming = _seed(db_session)
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    rows = db_session.query(TeamFeatures).filter_by(match_id=upcoming.id).all()
    assert len(rows) == 2
    teams = {r.team_id for r in rows}
    assert teams == {a.id, b.id}


def test_features_have_expected_ranges(db_session):
    a, b, upcoming = _seed(db_session)
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    row_a = db_session.query(TeamFeatures).filter_by(team_id=a.id, match_id=upcoming.id).one()
    assert 0.0 <= row_a.win_rate_recent_decayed <= 1.0
    assert 0.0 <= row_a.head_to_head_decayed <= 1.0
    assert 0.0 <= row_a.sos_score <= 1.0
    assert "de_mirage" in row_a.map_stats
    assert row_a.hltv_ranking_snapshot == 5


def test_idempotent_recompute(db_session):
    _seed(db_session)
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    compute_features_for_scheduled_matches(db_session)
    db_session.commit()
    assert db_session.query(TeamFeatures).count() == 2
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/features/test_compute.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/features/compute.py`**

```python
from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchResult,
    MatchStatus,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.features.head_to_head import compute_head_to_head
from cs2_predictor.pipeline.features.map_stats import compute_map_stats
from cs2_predictor.pipeline.features.recent_form import compute_recent_form
from cs2_predictor.pipeline.features.sos import compute_sos


def _finished_matches_for_team(session: Session, team_id: int, before: datetime) -> list[dict]:
    stmt = (
        select(Match, MatchResult)
        .join(MatchResult, MatchResult.match_id == Match.id)
        .where(Match.status == MatchStatus.FINISHED)
        .where(or_(Match.team_a_id == team_id, Match.team_b_id == team_id))
        .where(MatchResult.played_at < before)
    )
    rows = session.execute(stmt).all()
    output = []
    for match, result in rows:
        opponent_id = match.team_b_id if match.team_a_id == team_id else match.team_a_id
        opponent = session.get(Team, opponent_id)
        opponent_rank = opponent.hltv_ranking if opponent else None
        for map_name, _score in (result.score_detail or {}).items():
            output.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": map_name,
                "opponent_rank": opponent_rank,
                "opponent_id": opponent_id,
            })
        if not (result.score_detail or {}):
            output.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": None,
                "opponent_rank": opponent_rank,
                "opponent_id": opponent_id,
            })
    return output


def _team_features(
    session: Session, team_id: int, opponent_id: int, match: Match
) -> TeamFeatures:
    ref = match.scheduled_at
    history = _finished_matches_for_team(session, team_id, before=ref)
    h2h = [m for m in history if m["opponent_id"] == opponent_id]
    h2h_dicts = [{"played_at": m["played_at"], "team_won": m["won"]} for m in h2h]
    map_matches = [{"map": m["map"], "won": m["won"]} for m in history if m["map"]]

    team = session.get(Team, team_id)
    existing = (
        session.query(TeamFeatures)
        .filter_by(team_id=team_id, match_id=match.id)
        .one_or_none()
    )
    if existing is None:
        existing = TeamFeatures(team_id=team_id, match_id=match.id)
        session.add(existing)

    existing.win_rate_recent_decayed = compute_recent_form(
        [{"played_at": m["played_at"], "won": m["won"]} for m in history],
        reference_date=ref,
    )
    existing.head_to_head_decayed = compute_head_to_head(h2h_dicts, reference_date=ref)
    existing.hltv_ranking_snapshot = team.hltv_ranking if team else None
    existing.sos_score = compute_sos(
        [{"opponent_rank": m["opponent_rank"], "won": m["won"]} for m in history]
    )
    existing.map_stats = compute_map_stats(map_matches, map_pool=match.map_pool or [])
    existing.computed_at = datetime.now(timezone.utc)
    return existing


def compute_features_for_scheduled_matches(session: Session) -> int:
    matches = session.query(Match).filter(Match.status == MatchStatus.SCHEDULED).all()
    count = 0
    for match in matches:
        _team_features(session, match.team_a_id, match.team_b_id, match)
        _team_features(session, match.team_b_id, match.team_a_id, match)
        count += 1
    return count
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/features/test_compute.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/features/compute.py backend/tests/pipeline/features/test_compute.py
git commit -m "feat(features): compute and persist per-match team features"
```

---

## Phase 2: Pipeline — Modelo

### Task 11: Treinamento da regressão logística com validação temporal

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/model/__init__.py`
- Create: `backend/src/cs2_predictor/pipeline/model/dataset.py`
- Create: `backend/src/cs2_predictor/pipeline/model/train.py`
- Create: `backend/tests/pipeline/model/__init__.py`
- Create: `backend/tests/pipeline/model/test_train.py`

- [ ] **Step 1: Criar dirs e teste — `backend/tests/pipeline/model/test_train.py`**

```bash
mkdir -p backend/src/cs2_predictor/pipeline/model backend/tests/pipeline/model
touch backend/src/cs2_predictor/pipeline/model/__init__.py backend/tests/pipeline/model/__init__.py
```

```python
import numpy as np

from cs2_predictor.pipeline.model.train import (
    TrainedModel,
    train_logistic_regression,
)


def test_trained_model_predicts_probabilities():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = ["f1", "f2", "f3", "f4"]
    model = train_logistic_regression(X, y, feature_names=feature_names)
    assert isinstance(model, TrainedModel)
    probs = model.predict_proba(X[:5])
    assert probs.shape == (5,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_accuracy_is_reported():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(int)
    model = train_logistic_regression(X, y, feature_names=["a", "b", "c"])
    assert model.accuracy >= 0.7
    assert model.feature_names == ["a", "b", "c"]
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/model/test_train.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/model/train.py`**

```python
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class TrainedModel:
    estimator: LogisticRegression
    feature_names: list[str]
    accuracy: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of class 1 (team A wins)."""
        return self.estimator.predict_proba(X)[:, 1]


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    """Train logistic regression with held-out validation split.

    Caller is responsible for temporal ordering — to avoid data leakage,
    pass X/y already sorted by time and let train_test_split use shuffle=False
    via the explicit kwarg below.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False,
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = float(clf.score(X_val, y_val))
    return TrainedModel(estimator=clf, feature_names=feature_names, accuracy=accuracy)
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/model/test_train.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/model/ backend/tests/pipeline/model/
git commit -m "feat(model): train logistic regression with temporal validation"
```

---

### Task 12: Calibração com Platt Scaling

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/model/calibration.py`
- Create: `backend/tests/pipeline/model/test_calibration.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/model/test_calibration.py`**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from cs2_predictor.pipeline.model.calibration import CalibratedModel, calibrate_platt
from cs2_predictor.pipeline.model.train import TrainedModel


def _trained_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    return TrainedModel(estimator=clf, feature_names=["a", "b"], accuracy=0.9), X, y


def test_calibration_returns_probabilities_in_range():
    model, X, y = _trained_model()
    calibrated = calibrate_platt(model, X, y)
    assert isinstance(calibrated, CalibratedModel)
    probs = calibrated.predict_proba(X[:10])
    assert ((probs >= 0) & (probs <= 1)).all()


def test_calibration_preserves_feature_names():
    model, X, y = _trained_model()
    calibrated = calibrate_platt(model, X, y)
    assert calibrated.feature_names == ["a", "b"]
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/model/test_calibration.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/model/calibration.py`**

```python
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from cs2_predictor.pipeline.model.train import TrainedModel


@dataclass
class CalibratedModel:
    calibrator: CalibratedClassifierCV
    feature_names: list[str]
    base_accuracy: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.predict_proba(X)[:, 1]


def calibrate_platt(model: TrainedModel, X: np.ndarray, y: np.ndarray) -> CalibratedModel:
    """Apply Platt Scaling (sigmoid) on top of the trained estimator."""
    calibrator = CalibratedClassifierCV(model.estimator, method="sigmoid", cv="prefit")
    calibrator.fit(X, y)
    return CalibratedModel(
        calibrator=calibrator,
        feature_names=model.feature_names,
        base_accuracy=model.accuracy,
    )
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/model/test_calibration.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/model/calibration.py backend/tests/pipeline/model/test_calibration.py
git commit -m "feat(model): add Platt Scaling calibration"
```

---

### Task 13: Builder do dataset a partir do banco

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/model/dataset.py`
- Create: `backend/tests/pipeline/model/test_dataset.py`

Constrói matrizes `X, y` a partir das tabelas `team_features`, `matches`, `match_results`. Cada linha representa uma partida finalizada com features dos dois times concatenadas.

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/model/test_dataset.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.model.dataset import build_training_dataset


def _seed_finished_match(db_session, hltv_id, winner_is_a=True):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=hltv_id * 10 + 1, name=f"A{hltv_id}", hltv_ranking=5)
    b = Team(hltv_id=hltv_id * 10 + 2, name=f"B{hltv_id}", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=hltv_id, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=True, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now - timedelta(days=hltv_id),
        status=MatchStatus.FINISHED,
    )
    db_session.add(match)
    db_session.flush()
    db_session.add_all([
        TeamFeatures(team_id=a.id, match_id=match.id,
                     win_rate_recent_decayed=0.7, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=5, sos_score=0.6,
                     map_stats={"de_mirage": 0.75}),
        TeamFeatures(team_id=b.id, match_id=match.id,
                     win_rate_recent_decayed=0.4, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=20, sos_score=0.45,
                     map_stats={"de_mirage": 0.5}),
        MatchResult(match_id=match.id,
                    winner_id=a.id if winner_is_a else b.id,
                    score_detail={"de_mirage": [16, 12]},
                    played_at=now - timedelta(days=hltv_id)),
    ])
    db_session.commit()
    return match


def test_dataset_has_one_row_per_finished_match(db_session):
    _seed_finished_match(db_session, 1)
    _seed_finished_match(db_session, 2, winner_is_a=False)
    X, y, feature_names = build_training_dataset(db_session)
    assert X.shape[0] == 2
    assert y.tolist() in ([1, 0], [0, 1])
    assert "team_a_win_rate_recent_decayed" in feature_names
    assert "team_b_win_rate_recent_decayed" in feature_names


def test_dataset_is_ordered_by_played_at(db_session):
    _seed_finished_match(db_session, 5)
    _seed_finished_match(db_session, 1)
    X, y, _ = build_training_dataset(db_session)
    assert X.shape[0] == 2
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/model/test_dataset.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/model/dataset.py`**

```python
import numpy as np
from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchResult,
    MatchStatus,
    TeamFeatures,
)

SCALAR_FEATURES = [
    "win_rate_recent_decayed",
    "head_to_head_decayed",
    "hltv_ranking_snapshot",
    "sos_score",
]
CONTEXT_FEATURES = ["is_lan", "format_bo1", "format_bo3", "format_bo5"]


def _feature_vector(tf_a: TeamFeatures, tf_b: TeamFeatures, match: Match) -> list[float]:
    row: list[float] = []
    for name in SCALAR_FEATURES:
        row.append(float(getattr(tf_a, name) or 0.0))
    for name in SCALAR_FEATURES:
        row.append(float(getattr(tf_b, name) or 0.0))
    row.append(1.0 if match.is_lan else 0.0)
    row.append(1.0 if match.format.value == "BO1" else 0.0)
    row.append(1.0 if match.format.value == "BO3" else 0.0)
    row.append(1.0 if match.format.value == "BO5" else 0.0)
    return row


def feature_names() -> list[str]:
    names = [f"team_a_{n}" for n in SCALAR_FEATURES]
    names += [f"team_b_{n}" for n in SCALAR_FEATURES]
    names += CONTEXT_FEATURES
    return names


def build_training_dataset(session: Session) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X, y, feature_names from finished matches ordered by played_at."""
    rows = (
        session.query(Match, MatchResult)
        .join(MatchResult, MatchResult.match_id == Match.id)
        .filter(Match.status == MatchStatus.FINISHED)
        .order_by(MatchResult.played_at.asc())
        .all()
    )
    X_list, y_list = [], []
    for match, result in rows:
        tf_a = (
            session.query(TeamFeatures)
            .filter_by(team_id=match.team_a_id, match_id=match.id)
            .one_or_none()
        )
        tf_b = (
            session.query(TeamFeatures)
            .filter_by(team_id=match.team_b_id, match_id=match.id)
            .one_or_none()
        )
        if tf_a is None or tf_b is None:
            continue
        X_list.append(_feature_vector(tf_a, tf_b, match))
        y_list.append(1 if result.winner_id == match.team_a_id else 0)

    X = np.array(X_list, dtype=float) if X_list else np.empty((0, len(feature_names())))
    y = np.array(y_list, dtype=int)
    return X, y, feature_names()
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/model/test_dataset.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/model/dataset.py backend/tests/pipeline/model/test_dataset.py
git commit -m "feat(model): build training dataset from finished matches"
```

---

### Task 14: Gerar previsões para partidas agendadas

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/model/predict.py`
- Create: `backend/tests/pipeline/model/test_predict.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/model/test_predict.py`**

```python
from datetime import datetime, timedelta, timezone

import numpy as np

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchStatus,
    ModelRun,
    Prediction,
    Team,
    TeamFeatures,
)
from cs2_predictor.pipeline.model.predict import generate_predictions


class FakeCalibrated:
    feature_names = ["x"]
    base_accuracy = 0.7

    def predict_proba(self, X):
        return np.full(X.shape[0], 0.65)


def test_generates_prediction_per_scheduled_match(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=False, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.flush()
    db_session.add_all([
        TeamFeatures(team_id=a.id, match_id=match.id,
                     win_rate_recent_decayed=0.7, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=5, sos_score=0.6, map_stats={}),
        TeamFeatures(team_id=b.id, match_id=match.id,
                     win_rate_recent_decayed=0.4, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=20, sos_score=0.45, map_stats={}),
    ])
    db_session.add(ModelRun(version="v1", trained_at=now, accuracy=0.7,
                            features_used=["x"]))
    db_session.commit()

    count = generate_predictions(db_session, FakeCalibrated(), version="v1")
    db_session.commit()
    assert count == 1
    pred = db_session.query(Prediction).one()
    assert abs(pred.team_a_win_prob - 0.65) < 1e-6
    assert abs(pred.team_b_win_prob - 0.35) < 1e-6
    assert pred.model_version == "v1"


def test_skips_match_without_features(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A")
    b = Team(hltv_id=2, name="B")
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=[],
        tournament="x", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.commit()

    count = generate_predictions(db_session, FakeCalibrated(), version="v1")
    db_session.commit()
    assert count == 0
    assert db_session.query(Prediction).count() == 0
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/model/test_predict.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/model/predict.py`**

```python
import numpy as np
from sqlalchemy.orm import Session

from cs2_predictor.db.models import (
    Match,
    MatchStatus,
    Prediction,
    TeamFeatures,
)
from cs2_predictor.pipeline.model.dataset import _feature_vector


def generate_predictions(session: Session, model, version: str) -> int:
    """Generate and persist predictions for every scheduled match that has features.

    Existing predictions for the same match are replaced.
    """
    matches = session.query(Match).filter(Match.status == MatchStatus.SCHEDULED).all()
    count = 0
    for match in matches:
        tf_a = (
            session.query(TeamFeatures)
            .filter_by(team_id=match.team_a_id, match_id=match.id)
            .one_or_none()
        )
        tf_b = (
            session.query(TeamFeatures)
            .filter_by(team_id=match.team_b_id, match_id=match.id)
            .one_or_none()
        )
        if tf_a is None or tf_b is None:
            continue
        X = np.array([_feature_vector(tf_a, tf_b, match)], dtype=float)
        prob_a = float(model.predict_proba(X)[0])
        prob_b = 1.0 - prob_a

        existing = (
            session.query(Prediction)
            .filter_by(match_id=match.id, model_version=version)
            .one_or_none()
        )
        if existing is None:
            existing = Prediction(match_id=match.id, model_version=version)
            session.add(existing)
        existing.team_a_win_prob = prob_a
        existing.team_b_win_prob = prob_b
        existing.calibrated = True
        count += 1
    return count
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/model/test_predict.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/model/predict.py backend/tests/pipeline/model/test_predict.py
git commit -m "feat(model): generate calibrated predictions for scheduled matches"
```

---

### Task 15: Entry point do pipeline (`run.py`)

**Files:**
- Create: `backend/src/cs2_predictor/pipeline/run.py`
- Create: `backend/tests/pipeline/test_run.py`

- [ ] **Step 1: Escrever teste — `backend/tests/pipeline/test_run.py`**

```python
from unittest.mock import MagicMock, patch

from cs2_predictor.pipeline.run import run_pipeline


def test_run_pipeline_continues_on_scraper_failure(db_session):
    scraper = MagicMock()
    scraper.fetch_team_ranking.side_effect = Exception("HLTV down")
    with patch("cs2_predictor.pipeline.run.HLTVScraper", return_value=scraper):
        result = run_pipeline(session=db_session)
    assert result["status"] == "partial"
    assert "scraper_error" in result["errors"]


def test_run_pipeline_skips_retraining_below_threshold(db_session):
    scraper = MagicMock()
    scraper.fetch_team_ranking.return_value = []
    scraper.fetch_upcoming_matches.return_value = []
    scraper.fetch_match_results.return_value = []
    with patch("cs2_predictor.pipeline.run.HLTVScraper", return_value=scraper):
        result = run_pipeline(session=db_session, min_matches_to_retrain=10)
    assert result["retrained"] is False
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/pipeline/test_run.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/pipeline/run.py`**

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
from cs2_predictor.pipeline.scraper.hltv import HLTVScraper
from cs2_predictor.pipeline.scraper.persistence import (
    upsert_match_results,
    upsert_matches,
    upsert_teams,
)

logger = logging.getLogger(__name__)


def _next_version() -> str:
    return f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def _latest_version(session: Session) -> str | None:
    row = session.query(ModelRun).order_by(ModelRun.trained_at.desc()).first()
    return row.version if row else None


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
        scraper = HLTVScraper(base_url=settings.hltv_api_base_url)
        try:
            upsert_teams(session, scraper.fetch_team_ranking())
            upsert_matches(session, scraper.fetch_upcoming_matches())
            upsert_match_results(session, scraper.fetch_match_results())
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

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/pipeline/test_run.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Rodar suite completa do pipeline**

Run: `cd backend && uv run pytest tests/pipeline/ -v`
Expected: todos os testes do pipeline PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/src/cs2_predictor/pipeline/run.py backend/tests/pipeline/test_run.py
git commit -m "feat(pipeline): orchestrator with error isolation per stage"
```

---

## Phase 3: API (FastAPI)

### Task 16: Setup do app FastAPI + schemas Pydantic

**Files:**
- Create: `backend/src/cs2_predictor/api/__init__.py`
- Create: `backend/src/cs2_predictor/api/main.py`
- Create: `backend/src/cs2_predictor/api/schemas.py`
- Create: `backend/tests/api/__init__.py`
- Create: `backend/tests/api/conftest.py`
- Create: `backend/tests/api/test_health.py`

- [ ] **Step 1: Criar dirs**

```bash
mkdir -p backend/src/cs2_predictor/api backend/tests/api
touch backend/src/cs2_predictor/api/__init__.py backend/tests/api/__init__.py
```

- [ ] **Step 2: Criar `backend/src/cs2_predictor/api/schemas.py`**

```python
from datetime import datetime

from pydantic import BaseModel


class TeamSummary(BaseModel):
    id: int
    hltv_id: int
    name: str
    country: str | None
    hltv_ranking: int | None


class MatchPrediction(BaseModel):
    match_id: int
    hltv_match_id: int
    team_a: TeamSummary
    team_b: TeamSummary
    team_a_win_prob: float
    team_b_win_prob: float
    format: str
    is_lan: bool
    scheduled_at: datetime
    tournament: str | None
    model_version: str


class MatchFeatureSet(BaseModel):
    team_id: int
    win_rate_recent_decayed: float
    head_to_head_decayed: float
    hltv_ranking_snapshot: int | None
    sos_score: float
    map_stats: dict[str, float]


class MatchDetail(BaseModel):
    prediction: MatchPrediction
    features: list[MatchFeatureSet]


class ModelAccuracyEntry(BaseModel):
    version: str
    trained_at: datetime
    accuracy: float
    features_used: list[str]
```

- [ ] **Step 3: Criar `backend/src/cs2_predictor/api/main.py`**

```python
from fastapi import FastAPI

from cs2_predictor.api.routers import matches, model, teams


def create_app() -> FastAPI:
    app = FastAPI(title="CS2 Win Predictor API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(matches.router)
    app.include_router(teams.router)
    app.include_router(model.router)
    return app


app = create_app()
```

- [ ] **Step 4: Criar stubs vazios dos routers (para o import funcionar agora; preencheremos nas próximas tasks)**

```bash
mkdir -p backend/src/cs2_predictor/api/routers
touch backend/src/cs2_predictor/api/routers/__init__.py
```

Criar `backend/src/cs2_predictor/api/routers/matches.py`:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/matches", tags=["matches"])
```

Criar `backend/src/cs2_predictor/api/routers/teams.py`:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/teams", tags=["teams"])
```

Criar `backend/src/cs2_predictor/api/routers/model.py`:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/model", tags=["model"])
```

- [ ] **Step 5: Criar `backend/tests/api/conftest.py`**

```python
import pytest
from fastapi.testclient import TestClient

from cs2_predictor.api.main import create_app
from cs2_predictor.db.session import get_db


@pytest.fixture
def client(db_session):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)
```

- [ ] **Step 6: Criar `backend/tests/api/test_health.py`**

```python
def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 7: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/api/test_health.py -v`
Expected: 1 PASSED.

- [ ] **Step 8: Commit**

```bash
git add backend/src/cs2_predictor/api/ backend/tests/api/
git commit -m "feat(api): bootstrap FastAPI app with health check and router stubs"
```

---

### Task 17: Endpoint `GET /matches/upcoming`

**Files:**
- Modify: `backend/src/cs2_predictor/api/routers/matches.py`
- Create: `backend/tests/api/test_matches_upcoming.py`

- [ ] **Step 1: Escrever teste — `backend/tests/api/test_matches_upcoming.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchStatus,
    ModelRun,
    Prediction,
    Team,
)


def _seed(db_session, with_prediction=True):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="Navi", country="UA", hltv_ranking=1)
    b = Team(hltv_id=2, name="FaZe", country="EU", hltv_ranking=2)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=True, map_pool=["de_mirage"],
        tournament="Major", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.flush()
    if with_prediction:
        db_session.add(ModelRun(version="v1", trained_at=now, accuracy=0.7,
                                features_used=["x"]))
        db_session.add(Prediction(match_id=match.id, team_a_win_prob=0.6,
                                  team_b_win_prob=0.4, model_version="v1"))
    db_session.commit()
    return match


def test_upcoming_returns_matches_with_predictions(client, db_session):
    _seed(db_session)
    response = client.get("/matches/upcoming")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["team_a"]["name"] == "Navi"
    assert data[0]["team_a_win_prob"] == 0.6
    assert data[0]["model_version"] == "v1"


def test_upcoming_omits_matches_without_prediction(client, db_session):
    _seed(db_session, with_prediction=False)
    response = client.get("/matches/upcoming")
    assert response.status_code == 200
    assert response.json() == []
```

- [ ] **Step 2: Rodar — falha (rota vazia, retorna 404)**

Run: `cd backend && uv run pytest tests/api/test_matches_upcoming.py -v`
Expected: FAIL — 404 ou route missing.

- [ ] **Step 3: Implementar router — substituir `backend/src/cs2_predictor/api/routers/matches.py`**

```python
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cs2_predictor.api.schemas import (
    MatchDetail,
    MatchFeatureSet,
    MatchPrediction,
    TeamSummary,
)
from cs2_predictor.db.models import (
    Match,
    MatchStatus,
    Prediction,
    TeamFeatures,
)
from cs2_predictor.db.session import get_db

router = APIRouter(prefix="/matches", tags=["matches"])


def _to_team_summary(team) -> TeamSummary:
    return TeamSummary(
        id=team.id, hltv_id=team.hltv_id, name=team.name,
        country=team.country, hltv_ranking=team.hltv_ranking,
    )


def _to_prediction_dto(match: Match, prediction: Prediction) -> MatchPrediction:
    return MatchPrediction(
        match_id=match.id,
        hltv_match_id=match.hltv_id,
        team_a=_to_team_summary(match.team_a),
        team_b=_to_team_summary(match.team_b),
        team_a_win_prob=prediction.team_a_win_prob,
        team_b_win_prob=prediction.team_b_win_prob,
        format=match.format.value,
        is_lan=match.is_lan,
        scheduled_at=match.scheduled_at,
        tournament=match.tournament,
        model_version=prediction.model_version,
    )


def _latest_prediction_subquery(db: Session):
    """Subquery returning the latest prediction timestamp per match_id."""
    from sqlalchemy import func

    return (
        db.query(
            Prediction.match_id.label("match_id"),
            func.max(Prediction.created_at).label("latest_created_at"),
        )
        .group_by(Prediction.match_id)
        .subquery()
    )


@router.get("/upcoming", response_model=list[MatchPrediction])
def list_upcoming(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    latest = _latest_prediction_subquery(db)
    rows = (
        db.query(Match, Prediction)
        .join(Prediction, Prediction.match_id == Match.id)
        .join(
            latest,
            (latest.c.match_id == Prediction.match_id)
            & (latest.c.latest_created_at == Prediction.created_at),
        )
        .filter(Match.status == MatchStatus.SCHEDULED)
        .filter(Match.scheduled_at >= now)
        .order_by(Match.scheduled_at.asc())
        .all()
    )
    return [_to_prediction_dto(m, p) for m, p in rows]
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/api/test_matches_upcoming.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/api/routers/matches.py backend/tests/api/test_matches_upcoming.py
git commit -m "feat(api): GET /matches/upcoming"
```

---

### Task 18: Endpoints `GET /matches/{id}/prediction` e `/matches/{id}/features`

**Files:**
- Modify: `backend/src/cs2_predictor/api/routers/matches.py`
- Create: `backend/tests/api/test_matches_detail.py`

- [ ] **Step 1: Escrever teste — `backend/tests/api/test_matches_detail.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchStatus,
    ModelRun,
    Prediction,
    Team,
    TeamFeatures,
)


def _seed_full(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=100, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO3, is_lan=False, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now + timedelta(days=1),
        status=MatchStatus.SCHEDULED,
    )
    db_session.add(match)
    db_session.flush()
    db_session.add_all([
        TeamFeatures(team_id=a.id, match_id=match.id,
                     win_rate_recent_decayed=0.7, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=5, sos_score=0.6,
                     map_stats={"de_mirage": 0.75}),
        TeamFeatures(team_id=b.id, match_id=match.id,
                     win_rate_recent_decayed=0.4, head_to_head_decayed=0.5,
                     hltv_ranking_snapshot=20, sos_score=0.45,
                     map_stats={"de_mirage": 0.5}),
        ModelRun(version="v1", trained_at=now, accuracy=0.7, features_used=["x"]),
        Prediction(match_id=match.id, team_a_win_prob=0.62,
                   team_b_win_prob=0.38, model_version="v1"),
    ])
    db_session.commit()
    return match


def test_prediction_endpoint_returns_full_detail(client, db_session):
    match = _seed_full(db_session)
    response = client.get(f"/matches/{match.id}/prediction")
    assert response.status_code == 200
    data = response.json()
    assert data["team_a_win_prob"] == 0.62
    assert data["tournament"] == "x"


def test_prediction_endpoint_404_when_missing(client):
    response = client.get("/matches/99999/prediction")
    assert response.status_code == 404


def test_features_endpoint_returns_both_teams(client, db_session):
    match = _seed_full(db_session)
    response = client.get(f"/matches/{match.id}/features")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    a_features = [f for f in data if f["win_rate_recent_decayed"] == 0.7][0]
    assert a_features["map_stats"] == {"de_mirage": 0.75}
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/api/test_matches_detail.py -v`
Expected: FAIL.

- [ ] **Step 3: Adicionar handlers ao `backend/src/cs2_predictor/api/routers/matches.py`**

Acrescentar no final do arquivo (depois do handler de `/upcoming`):

```python
@router.get("/{match_id}/prediction", response_model=MatchPrediction)
def get_prediction(match_id: int, db: Session = Depends(get_db)):
    row = (
        db.query(Match, Prediction)
        .join(Prediction, Prediction.match_id == Match.id)
        .filter(Match.id == match_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    match, prediction = row
    return _to_prediction_dto(match, prediction)


@router.get("/{match_id}/features", response_model=list[MatchFeatureSet])
def get_features(match_id: int, db: Session = Depends(get_db)):
    rows = db.query(TeamFeatures).filter_by(match_id=match_id).all()
    if not rows:
        raise HTTPException(status_code=404, detail="features not found")
    return [
        MatchFeatureSet(
            team_id=r.team_id,
            win_rate_recent_decayed=r.win_rate_recent_decayed,
            head_to_head_decayed=r.head_to_head_decayed,
            hltv_ranking_snapshot=r.hltv_ranking_snapshot,
            sos_score=r.sos_score,
            map_stats=r.map_stats,
        )
        for r in rows
    ]
```

- [ ] **Step 4: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/api/test_matches_detail.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/api/routers/matches.py backend/tests/api/test_matches_detail.py
git commit -m "feat(api): GET /matches/{id}/prediction and /features"
```

---

### Task 19: Endpoints de `/teams`

**Files:**
- Modify: `backend/src/cs2_predictor/api/routers/teams.py`
- Modify: `backend/src/cs2_predictor/api/schemas.py`
- Create: `backend/tests/api/test_teams.py`

- [ ] **Step 1: Adicionar schemas em `backend/src/cs2_predictor/api/schemas.py`** (acrescentar no final)

```python
class TeamStats(BaseModel):
    team: TeamSummary
    recent_form: float
    last_matches_played: int


class TeamDetail(BaseModel):
    team: TeamSummary
    recent_form: float
    map_winrates: dict[str, float]
    last_matches_played: int
```

- [ ] **Step 2: Escrever teste — `backend/tests/api/test_teams.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import (
    Match,
    MatchFormat,
    MatchResult,
    MatchStatus,
    Team,
)


def _seed_team_with_history(db_session):
    now = datetime.now(timezone.utc)
    a = Team(hltv_id=1, name="A", country="BR", hltv_ranking=5)
    b = Team(hltv_id=2, name="B", country="EU", hltv_ranking=20)
    db_session.add_all([a, b])
    db_session.flush()
    match = Match(
        hltv_id=99, team_a_id=a.id, team_b_id=b.id,
        format=MatchFormat.BO1, is_lan=False, map_pool=["de_mirage"],
        tournament="x", scheduled_at=now - timedelta(days=5),
        status=MatchStatus.FINISHED,
    )
    db_session.add(match)
    db_session.flush()
    db_session.add(MatchResult(
        match_id=match.id, winner_id=a.id,
        score_detail={"de_mirage": [16, 12]},
        played_at=now - timedelta(days=5),
    ))
    db_session.commit()
    return a


def test_list_teams(client, db_session):
    _seed_team_with_history(db_session)
    response = client.get("/teams")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert any(t["team"]["name"] == "A" for t in data)


def test_team_detail_returns_map_winrates(client, db_session):
    team = _seed_team_with_history(db_session)
    response = client.get(f"/teams/{team.id}/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["team"]["name"] == "A"
    assert "de_mirage" in data["map_winrates"]
    assert data["last_matches_played"] == 1


def test_team_detail_404_when_missing(client):
    response = client.get("/teams/99999/stats")
    assert response.status_code == 404
```

- [ ] **Step 3: Rodar — falha**

Run: `cd backend && uv run pytest tests/api/test_teams.py -v`
Expected: FAIL.

- [ ] **Step 4: Implementar `backend/src/cs2_predictor/api/routers/teams.py`**

```python
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from cs2_predictor.api.schemas import TeamDetail, TeamStats, TeamSummary
from cs2_predictor.db.models import (
    Match,
    MatchResult,
    MatchStatus,
    Team,
)
from cs2_predictor.db.session import get_db
from cs2_predictor.pipeline.features.map_stats import compute_map_stats
from cs2_predictor.pipeline.features.recent_form import compute_recent_form

router = APIRouter(prefix="/teams", tags=["teams"])


def _team_history(db: Session, team_id: int) -> list[dict]:
    rows = (
        db.query(Match, MatchResult)
        .join(MatchResult, MatchResult.match_id == Match.id)
        .filter(Match.status == MatchStatus.FINISHED)
        .filter(or_(Match.team_a_id == team_id, Match.team_b_id == team_id))
        .all()
    )
    history = []
    for match, result in rows:
        for map_name, _score in (result.score_detail or {}).items():
            history.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": map_name,
            })
        if not (result.score_detail or {}):
            history.append({
                "played_at": result.played_at,
                "won": result.winner_id == team_id,
                "map": None,
            })
    return history


def _to_summary(team: Team) -> TeamSummary:
    return TeamSummary(
        id=team.id, hltv_id=team.hltv_id, name=team.name,
        country=team.country, hltv_ranking=team.hltv_ranking,
    )


@router.get("", response_model=list[TeamStats])
def list_teams(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    teams = db.query(Team).order_by(Team.hltv_ranking.asc().nulls_last()).all()
    out: list[TeamStats] = []
    for team in teams:
        history = _team_history(db, team.id)
        out.append(TeamStats(
            team=_to_summary(team),
            recent_form=compute_recent_form(
                [{"played_at": h["played_at"], "won": h["won"]} for h in history],
                reference_date=now,
            ),
            last_matches_played=len({(h["played_at"]) for h in history}),
        ))
    return out


@router.get("/{team_id}/stats", response_model=TeamDetail)
def team_stats(team_id: int, db: Session = Depends(get_db)):
    team = db.get(Team, team_id)
    if team is None:
        raise HTTPException(status_code=404, detail="team not found")
    history = _team_history(db, team.id)
    map_history = [{"map": h["map"], "won": h["won"]} for h in history if h["map"]]
    all_maps = sorted({m["map"] for m in map_history})
    return TeamDetail(
        team=_to_summary(team),
        recent_form=compute_recent_form(
            [{"played_at": h["played_at"], "won": h["won"]} for h in history],
            reference_date=datetime.now(timezone.utc),
        ),
        map_winrates=compute_map_stats(map_history, map_pool=all_maps),
        last_matches_played=len({h["played_at"] for h in history}),
    )
```

- [ ] **Step 5: Rodar — deve passar**

Run: `cd backend && uv run pytest tests/api/test_teams.py -v`
Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add backend/src/cs2_predictor/api/routers/teams.py backend/src/cs2_predictor/api/schemas.py backend/tests/api/test_teams.py
git commit -m "feat(api): GET /teams and /teams/{id}/stats"
```

---

### Task 20: Endpoint `GET /model/accuracy`

**Files:**
- Modify: `backend/src/cs2_predictor/api/routers/model.py`
- Create: `backend/tests/api/test_model.py`

- [ ] **Step 1: Escrever teste — `backend/tests/api/test_model.py`**

```python
from datetime import datetime, timedelta, timezone

from cs2_predictor.db.models import ModelRun


def test_accuracy_lists_versions_newest_first(client, db_session):
    now = datetime.now(timezone.utc)
    db_session.add_all([
        ModelRun(version="v1", trained_at=now - timedelta(days=2),
                 accuracy=0.62, features_used=["x"]),
        ModelRun(version="v2", trained_at=now - timedelta(days=1),
                 accuracy=0.68, features_used=["x", "y"]),
        ModelRun(version="v3", trained_at=now,
                 accuracy=0.71, features_used=["x", "y", "z"]),
    ])
    db_session.commit()
    response = client.get("/model/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert [r["version"] for r in data] == ["v3", "v2", "v1"]
    assert data[0]["accuracy"] == 0.71


def test_accuracy_empty(client):
    response = client.get("/model/accuracy")
    assert response.status_code == 200
    assert response.json() == []
```

- [ ] **Step 2: Rodar — falha**

Run: `cd backend && uv run pytest tests/api/test_model.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementar `backend/src/cs2_predictor/api/routers/model.py`**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from cs2_predictor.api.schemas import ModelAccuracyEntry
from cs2_predictor.db.models import ModelRun
from cs2_predictor.db.session import get_db

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/accuracy", response_model=list[ModelAccuracyEntry])
def accuracy(db: Session = Depends(get_db)):
    rows = db.query(ModelRun).order_by(ModelRun.trained_at.desc()).all()
    return [
        ModelAccuracyEntry(
            version=r.version,
            trained_at=r.trained_at,
            accuracy=r.accuracy,
            features_used=r.features_used,
        )
        for r in rows
    ]
```

- [ ] **Step 4: Rodar suite completa da API**

Run: `cd backend && uv run pytest tests/api/ -v`
Expected: todos os testes da API PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/api/routers/model.py backend/tests/api/test_model.py
git commit -m "feat(api): GET /model/accuracy"
```

---

### Task 21: CORS para o frontend

**Files:**
- Modify: `backend/src/cs2_predictor/api/main.py`
- Modify: `backend/src/cs2_predictor/config.py`
- Modify: `backend/.env.example`

- [ ] **Step 1: Adicionar `cors_origins` em `backend/src/cs2_predictor/config.py`**

Substituir a classe Settings por:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    database_url_test: str = ""
    hltv_api_base_url: str = "http://localhost:8000"
    pipeline_interval_hours: int = 6
    min_matches_to_retrain: int = 10
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
```

- [ ] **Step 2: Atualizar `backend/src/cs2_predictor/api/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cs2_predictor.api.routers import matches, model, teams
from cs2_predictor.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="CS2 Win Predictor API")
    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(matches.router)
    app.include_router(teams.router)
    app.include_router(model.router)
    return app


app = create_app()
```

- [ ] **Step 3: Adicionar em `backend/.env.example`**

```
CORS_ORIGINS=http://localhost:3000
```

- [ ] **Step 4: Rodar testes** — devem continuar passando.

Run: `cd backend && uv run pytest -v`
Expected: tudo PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/src/cs2_predictor/api/main.py backend/src/cs2_predictor/config.py backend/.env.example
git commit -m "feat(api): enable CORS for frontend origin"
```

---

## Phase 4: Frontend (Next.js 15)

### Task 22: Bootstrap do projeto Next.js + Tailwind + cliente HTTP

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.mjs`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.mjs`
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/globals.css`
- Create: `frontend/app/page.tsx` (placeholder)
- Create: `frontend/lib/api.ts`
- Create: `frontend/lib/types.ts`
- Create: `frontend/.env.local.example`

- [ ] **Step 1: Inicializar o projeto Next.js manualmente**

```bash
mkdir -p frontend/app frontend/components frontend/lib
cd frontend
```

Criar `frontend/package.json`:

```json
{
  "name": "cs2-predictor-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "next": "^15.0.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.5.0"
  }
}
```

- [ ] **Step 2: Criar `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Criar `frontend/next.config.mjs`**

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};
export default nextConfig;
```

- [ ] **Step 4: Criar `frontend/tailwind.config.ts`**

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
export default config;
```

- [ ] **Step 5: Criar `frontend/postcss.config.mjs`**

```javascript
const config = { plugins: { tailwindcss: {}, autoprefixer: {} } };
export default config;
```

- [ ] **Step 6: Criar `frontend/app/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root { color-scheme: dark; }
body { @apply bg-zinc-950 text-zinc-100; }
```

- [ ] **Step 7: Criar `frontend/app/layout.tsx`**

```tsx
import "./globals.css";
import type { Metadata, Viewport } from "next";

export const metadata: Metadata = {
  title: "CS2 Win Predictor",
  description: "Probabilidade estatística de vitória em partidas profissionais de CS2.",
};

export const viewport: Viewport = { themeColor: "#0a0a0a" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="min-h-screen">
        <header className="border-b border-zinc-800 px-6 py-4">
          <a href="/" className="text-lg font-semibold">CS2 Win Predictor</a>
          <nav className="ml-6 inline-flex gap-4 text-sm text-zinc-400">
            <a href="/teams" className="hover:text-zinc-100">Times</a>
            <a href="/model" className="hover:text-zinc-100">Modelo</a>
          </nav>
        </header>
        <main className="px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
```

- [ ] **Step 8: Criar `frontend/lib/types.ts`**

```typescript
export type TeamSummary = {
  id: number;
  hltv_id: number;
  name: string;
  country: string | null;
  hltv_ranking: number | null;
};

export type MatchPrediction = {
  match_id: number;
  hltv_match_id: number;
  team_a: TeamSummary;
  team_b: TeamSummary;
  team_a_win_prob: number;
  team_b_win_prob: number;
  format: "BO1" | "BO3" | "BO5";
  is_lan: boolean;
  scheduled_at: string;
  tournament: string | null;
  model_version: string;
};

export type MatchFeatureSet = {
  team_id: number;
  win_rate_recent_decayed: number;
  head_to_head_decayed: number;
  hltv_ranking_snapshot: number | null;
  sos_score: number;
  map_stats: Record<string, number>;
};

export type TeamStats = {
  team: TeamSummary;
  recent_form: number;
  last_matches_played: number;
};

export type TeamDetail = {
  team: TeamSummary;
  recent_form: number;
  map_winrates: Record<string, number>;
  last_matches_played: number;
};

export type ModelAccuracyEntry = {
  version: string;
  trained_at: string;
  accuracy: number;
  features_used: string[];
};
```

- [ ] **Step 9: Criar `frontend/lib/api.ts`**

```typescript
import type {
  MatchFeatureSet,
  MatchPrediction,
  ModelAccuracyEntry,
  TeamDetail,
  TeamStats,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    next: { revalidate: 60 },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`API ${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  upcomingMatches: () => fetchJson<MatchPrediction[]>("/matches/upcoming"),
  matchPrediction: (id: number) => fetchJson<MatchPrediction>(`/matches/${id}/prediction`),
  matchFeatures: (id: number) => fetchJson<MatchFeatureSet[]>(`/matches/${id}/features`),
  listTeams: () => fetchJson<TeamStats[]>("/teams"),
  teamDetail: (id: number) => fetchJson<TeamDetail>(`/teams/${id}/stats`),
  modelAccuracy: () => fetchJson<ModelAccuracyEntry[]>("/model/accuracy"),
};
```

- [ ] **Step 10: Criar placeholder `frontend/app/page.tsx`**

```tsx
export default function HomePage() {
  return <p>CS2 Predictor — em construção.</p>;
}
```

- [ ] **Step 11: Criar `frontend/.env.local.example`**

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 12: Instalar deps e validar build**

```bash
cd frontend && pnpm install && pnpm typecheck && pnpm build
```

Expected: build sem erros.

- [ ] **Step 13: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): bootstrap Next.js 15 with Tailwind and API client"
```

---

### Task 23: Componentes reutilizáveis (`ProbabilityBar`, `MatchCard`)

**Files:**
- Create: `frontend/components/ProbabilityBar.tsx`
- Create: `frontend/components/MatchCard.tsx`

- [ ] **Step 1: Criar `frontend/components/ProbabilityBar.tsx`**

```tsx
type Props = {
  leftLabel: string;
  rightLabel: string;
  leftProb: number;
};

export function ProbabilityBar({ leftLabel, rightLabel, leftProb }: Props) {
  const leftPct = Math.round(leftProb * 100);
  const rightPct = 100 - leftPct;
  return (
    <div className="w-full">
      <div className="mb-1 flex justify-between text-xs text-zinc-400">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded bg-zinc-800">
        <div className="bg-emerald-500" style={{ width: `${leftPct}%` }} />
        <div className="bg-rose-500" style={{ width: `${rightPct}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-sm font-medium">
        <span>{leftPct}%</span>
        <span>{rightPct}%</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Criar `frontend/components/MatchCard.tsx`**

```tsx
import Link from "next/link";

import type { MatchPrediction } from "@/lib/types";
import { ProbabilityBar } from "./ProbabilityBar";

function formatDate(iso: string) {
  return new Date(iso).toLocaleString("pt-BR", {
    day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

export function MatchCard({ match }: { match: MatchPrediction }) {
  return (
    <Link
      href={`/matches/${match.match_id}`}
      className="block rounded-lg border border-zinc-800 bg-zinc-900 p-4 hover:border-zinc-700"
    >
      <div className="mb-2 flex items-center justify-between text-xs text-zinc-400">
        <span>{match.tournament ?? "—"}</span>
        <span>{match.format} {match.is_lan ? "· LAN" : "· Online"}</span>
      </div>
      <div className="mb-3 flex items-center justify-between font-semibold">
        <span>{match.team_a.name}</span>
        <span className="text-xs text-zinc-500">{formatDate(match.scheduled_at)}</span>
        <span>{match.team_b.name}</span>
      </div>
      <ProbabilityBar
        leftLabel={match.team_a.name}
        rightLabel={match.team_b.name}
        leftProb={match.team_a_win_prob}
      />
    </Link>
  );
}
```

- [ ] **Step 3: Validar typecheck**

Run: `cd frontend && pnpm typecheck`
Expected: sem erros.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/
git commit -m "feat(frontend): add ProbabilityBar and MatchCard components"
```

---

### Task 24: Página `/` (dashboard de próximas partidas)

**Files:**
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Substituir `frontend/app/page.tsx`**

```tsx
import { api } from "@/lib/api";
import { MatchCard } from "@/components/MatchCard";

export const revalidate = 60;

export default async function HomePage() {
  let matches: Awaited<ReturnType<typeof api.upcomingMatches>> = [];
  let errored = false;
  try {
    matches = await api.upcomingMatches();
  } catch {
    errored = true;
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-6 text-2xl font-semibold">Próximas partidas</h1>
      {errored && (
        <p className="rounded border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-200">
          Não foi possível carregar as partidas no momento.
        </p>
      )}
      {!errored && matches.length === 0 && (
        <p className="text-zinc-400">Nenhuma partida agendada com previsão disponível.</p>
      )}
      <div className="space-y-3">
        {matches.map((m) => <MatchCard key={m.match_id} match={m} />)}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Validar build**

Run: `cd frontend && pnpm build`
Expected: build sem erros.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat(frontend): dashboard with upcoming matches"
```

---

### Task 25: Página de detalhe `/matches/[id]`

**Files:**
- Create: `frontend/app/matches/[id]/page.tsx`

- [ ] **Step 1: Criar `frontend/app/matches/[id]/page.tsx`**

```tsx
import { notFound } from "next/navigation";

import { api } from "@/lib/api";
import { ProbabilityBar } from "@/components/ProbabilityBar";

export const revalidate = 60;

type FeatureRowProps = { label: string; value: number | string | null };
function FeatureRow({ label, value }: FeatureRowProps) {
  return (
    <div className="flex justify-between border-b border-zinc-800 py-1 text-sm">
      <span className="text-zinc-400">{label}</span>
      <span>{value ?? "—"}</span>
    </div>
  );
}

export default async function MatchDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const matchId = Number(id);
  if (Number.isNaN(matchId)) notFound();

  let prediction;
  let features;
  try {
    [prediction, features] = await Promise.all([
      api.matchPrediction(matchId),
      api.matchFeatures(matchId),
    ]);
  } catch {
    notFound();
  }

  const teamA = features.find((f) => f.team_id === prediction.team_a.id);
  const teamB = features.find((f) => f.team_id === prediction.team_b.id);

  return (
    <div className="mx-auto max-w-3xl">
      <p className="mb-1 text-sm text-zinc-400">
        {prediction.tournament ?? "—"} · {prediction.format} ·{" "}
        {prediction.is_lan ? "LAN" : "Online"}
      </p>
      <h1 className="mb-6 text-2xl font-semibold">
        {prediction.team_a.name} <span className="text-zinc-500">vs</span> {prediction.team_b.name}
      </h1>
      <div className="mb-8">
        <ProbabilityBar
          leftLabel={prediction.team_a.name}
          rightLabel={prediction.team_b.name}
          leftProb={prediction.team_a_win_prob}
        />
        <p className="mt-2 text-xs text-zinc-500">
          Modelo {prediction.model_version} · agendado para{" "}
          {new Date(prediction.scheduled_at).toLocaleString("pt-BR")}
        </p>
      </div>

      <h2 className="mb-3 text-lg font-semibold">Features</h2>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {[
          { team: prediction.team_a, feats: teamA },
          { team: prediction.team_b, feats: teamB },
        ].map(({ team, feats }) => (
          <div key={team.id} className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="mb-2 font-semibold">{team.name}</h3>
            <FeatureRow label="Forma recente (decay)" value={feats?.win_rate_recent_decayed.toFixed(2) ?? null} />
            <FeatureRow label="H2H (decay)" value={feats?.head_to_head_decayed.toFixed(2) ?? null} />
            <FeatureRow label="Ranking HLTV (snapshot)" value={feats?.hltv_ranking_snapshot ?? null} />
            <FeatureRow label="SOS" value={feats?.sos_score.toFixed(2) ?? null} />
            {feats && Object.entries(feats.map_stats).map(([map, rate]) => (
              <FeatureRow key={map} label={`Win rate ${map}`} value={rate.toFixed(2)} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Validar build**

Run: `cd frontend && pnpm build`
Expected: build sem erros.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/matches/
git commit -m "feat(frontend): match detail page with features breakdown"
```

---

### Task 26: Página `/teams` (ranking de times)

**Files:**
- Create: `frontend/app/teams/page.tsx`

- [ ] **Step 1: Criar `frontend/app/teams/page.tsx`**

```tsx
import Link from "next/link";

import { api } from "@/lib/api";

export const revalidate = 300;

export default async function TeamsPage() {
  let teams: Awaited<ReturnType<typeof api.listTeams>> = [];
  try {
    teams = await api.listTeams();
  } catch {
    return <p className="text-rose-300">Não foi possível carregar os times.</p>;
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-6 text-2xl font-semibold">Times</h1>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-zinc-400">
            <th className="py-2">Ranking</th>
            <th className="py-2">Time</th>
            <th className="py-2">País</th>
            <th className="py-2">Forma recente</th>
            <th className="py-2">Partidas</th>
          </tr>
        </thead>
        <tbody>
          {teams.map((t) => (
            <tr key={t.team.id} className="border-t border-zinc-800">
              <td className="py-2">{t.team.hltv_ranking ?? "—"}</td>
              <td className="py-2">
                <Link href={`/teams/${t.team.id}`} className="hover:text-emerald-400">
                  {t.team.name}
                </Link>
              </td>
              <td className="py-2 text-zinc-400">{t.team.country ?? "—"}</td>
              <td className="py-2">{(t.recent_form * 100).toFixed(0)}%</td>
              <td className="py-2 text-zinc-400">{t.last_matches_played}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: Validar build**

Run: `cd frontend && pnpm build`
Expected: build sem erros.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/teams/page.tsx
git commit -m "feat(frontend): teams ranking page"
```

---

### Task 27: Página `/teams/[id]` (perfil do time)

**Files:**
- Create: `frontend/app/teams/[id]/page.tsx`

- [ ] **Step 1: Criar `frontend/app/teams/[id]/page.tsx`**

```tsx
import { notFound } from "next/navigation";

import { api } from "@/lib/api";

export const revalidate = 300;

export default async function TeamDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const teamId = Number(id);
  if (Number.isNaN(teamId)) notFound();

  let detail;
  try {
    detail = await api.teamDetail(teamId);
  } catch {
    notFound();
  }

  return (
    <div className="mx-auto max-w-3xl">
      <p className="mb-1 text-sm text-zinc-400">
        {detail.team.country ?? "—"} · Ranking HLTV {detail.team.hltv_ranking ?? "—"}
      </p>
      <h1 className="mb-6 text-2xl font-semibold">{detail.team.name}</h1>

      <div className="mb-8 grid grid-cols-2 gap-4">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-400">Forma recente</p>
          <p className="text-2xl font-semibold">{(detail.recent_form * 100).toFixed(0)}%</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-400">Partidas registradas</p>
          <p className="text-2xl font-semibold">{detail.last_matches_played}</p>
        </div>
      </div>

      <h2 className="mb-3 text-lg font-semibold">Win rate por mapa</h2>
      <div className="space-y-1">
        {Object.entries(detail.map_winrates).map(([map, rate]) => (
          <div key={map} className="flex items-center justify-between border-b border-zinc-800 py-1 text-sm">
            <span className="text-zinc-400">{map}</span>
            <span>{(rate * 100).toFixed(0)}%</span>
          </div>
        ))}
        {Object.keys(detail.map_winrates).length === 0 && (
          <p className="text-zinc-400">Sem histórico por mapa ainda.</p>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Validar build**

Run: `cd frontend && pnpm build`
Expected: build sem erros.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/teams/
git commit -m "feat(frontend): team detail page with map winrates"
```

---

### Task 28: Página `/model` (transparência do modelo)

**Files:**
- Create: `frontend/app/model/page.tsx`

- [ ] **Step 1: Criar `frontend/app/model/page.tsx`**

```tsx
import { api } from "@/lib/api";

export const revalidate = 600;

export default async function ModelPage() {
  let runs: Awaited<ReturnType<typeof api.modelAccuracy>> = [];
  try {
    runs = await api.modelAccuracy();
  } catch {
    return <p className="text-rose-300">Não foi possível carregar o histórico do modelo.</p>;
  }

  const current = runs[0];

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-2 text-2xl font-semibold">Transparência do modelo</h1>
      <p className="mb-6 text-sm text-zinc-400">
        Cada execução do pipeline retreina o modelo e registra a acurácia em validação temporal.
        As probabilidades são calibradas via Platt Scaling para refletir frequências reais.
      </p>

      {current && (
        <div className="mb-8 rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-400">Versão atual</p>
          <p className="text-xl font-semibold">{current.version}</p>
          <p className="mt-2 text-sm">
            Acurácia: <span className="font-semibold">{(current.accuracy * 100).toFixed(1)}%</span>
          </p>
          <p className="text-xs text-zinc-500">
            Treinado em {new Date(current.trained_at).toLocaleString("pt-BR")}
          </p>
          <p className="mt-2 text-xs text-zinc-400">
            Features: {current.features_used.join(", ")}
          </p>
        </div>
      )}

      <h2 className="mb-3 text-lg font-semibold">Histórico de versões</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-zinc-400">
            <th className="py-2">Versão</th>
            <th className="py-2">Treinado em</th>
            <th className="py-2">Acurácia</th>
            <th className="py-2"># Features</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.version} className="border-t border-zinc-800">
              <td className="py-2">{r.version}</td>
              <td className="py-2 text-zinc-400">
                {new Date(r.trained_at).toLocaleString("pt-BR")}
              </td>
              <td className="py-2">{(r.accuracy * 100).toFixed(1)}%</td>
              <td className="py-2 text-zinc-400">{r.features_used.length}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: Validar build completo**

Run: `cd frontend && pnpm build`
Expected: build sem erros para todas as rotas.

- [ ] **Step 3: Smoke test manual (executor opcional, recomendado)**

Em terminais separados:
- Terminal A: `cd backend && uv run uvicorn cs2_predictor.api.main:app --reload`
- Terminal B: `cd frontend && pnpm dev`
- Abrir `http://localhost:3000` e verificar que as 5 páginas renderizam (mesmo que com dados vazios).

- [ ] **Step 4: Commit**

```bash
git add frontend/app/model/
git commit -m "feat(frontend): model transparency page"
```

---

## Phase 5: Deployment

### Task 29: GitHub Action que roda o pipeline a cada 6h

**Files:**
- Create: `.github/workflows/pipeline.yml`

- [ ] **Step 1: Criar `.github/workflows/pipeline.yml`**

```yaml
name: Pipeline

on:
  schedule:
    - cron: "0 */6 * * *"
  workflow_dispatch: {}

concurrency:
  group: pipeline
  cancel-in-progress: false

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          uv venv
          uv pip install -e ".[dev]"

      - name: Apply migrations
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          uv run alembic upgrade head

      - name: Run pipeline
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          HLTV_API_BASE_URL: ${{ secrets.HLTV_API_BASE_URL }}
          PIPELINE_INTERVAL_HOURS: 6
          MIN_MATCHES_TO_RETRAIN: 10
        run: |
          export PATH="$HOME/.local/bin:$PATH"
          uv run python -m cs2_predictor.pipeline.run
```

- [ ] **Step 2: Documentar secrets necessários (output do plano, não execução automática)**

No repositório no GitHub (Settings → Secrets and variables → Actions), criar:
- `DATABASE_URL` — connection string Postgres (Neon)
- `HLTV_API_BASE_URL` — URL pública da instância `eupeutro/hltv-api` (executor deve subir uma própria ou usar serviço pago/público)

- [ ] **Step 3: Commit**

```bash
git add .github/
git commit -m "ci: schedule pipeline run every 6 hours"
```

---

### Task 30: Configuração de deploy do FastAPI no Render

**Files:**
- Create: `backend/render.yaml`
- Create: `backend/start.sh`

- [ ] **Step 1: Criar `backend/start.sh`**

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv run alembic upgrade head
exec uv run uvicorn cs2_predictor.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

Tornar executável: `chmod +x backend/start.sh`

- [ ] **Step 2: Criar `backend/render.yaml`** (blueprint)

```yaml
services:
  - type: web
    name: cs2-predictor-api
    env: python
    plan: free
    rootDir: backend
    buildCommand: |
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
      uv venv
      uv pip install -e ".[dev]"
    startCommand: bash start.sh
    envVars:
      - key: PYTHON_VERSION
        value: "3.12"
      - key: DATABASE_URL
        sync: false
      - key: HLTV_API_BASE_URL
        sync: false
      - key: CORS_ORIGINS
        sync: false
      - key: PATH
        value: "/opt/render/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
```

- [ ] **Step 3: Documentar deploy manual**

Sem execução automatizada — o usuário precisa:
1. Conectar o repo no Render (https://dashboard.render.com)
2. "New +" → "Blueprint" → apontar para `backend/render.yaml`
3. Definir manualmente `DATABASE_URL`, `HLTV_API_BASE_URL`, `CORS_ORIGINS` (URL do Vercel)

- [ ] **Step 4: Commit**

```bash
git add backend/render.yaml backend/start.sh
git commit -m "deploy: add Render blueprint and start script"
```

---

### Task 31: Configuração de deploy do frontend no Vercel

**Files:**
- Create: `frontend/vercel.json`
- Create: `frontend/.vercelignore`

- [ ] **Step 1: Criar `frontend/vercel.json`**

```json
{
  "framework": "nextjs",
  "buildCommand": "pnpm build",
  "installCommand": "pnpm install",
  "devCommand": "pnpm dev"
}
```

- [ ] **Step 2: Criar `frontend/.vercelignore`**

```
node_modules
.next
```

- [ ] **Step 3: Documentar deploy manual**

1. Em https://vercel.com → "Add New Project" → importar o repo
2. Root directory: `frontend/`
3. Environment variables: `NEXT_PUBLIC_API_URL` = URL pública do Render
4. Deploy

- [ ] **Step 4: Commit**

```bash
git add frontend/vercel.json frontend/.vercelignore
git commit -m "deploy: add Vercel project configuration"
```

---

### Task 32: README com instruções de setup

**Files:**
- Create: `README.md`

- [ ] **Step 1: Criar `README.md` na raiz**

````markdown
# CS2 Win Predictor

Plataforma que prevê probabilidade de vitória em partidas profissionais de Counter-Strike 2 usando regressão logística calibrada sobre dados do HLTV.

## Stack

- **Backend** (`backend/`): Python 3.12, FastAPI, SQLAlchemy, scikit-learn, Alembic, uv
- **Frontend** (`frontend/`): Next.js 15 (App Router), TypeScript, Tailwind, pnpm
- **Banco**: PostgreSQL (Neon na nuvem)
- **Pipeline**: GitHub Actions (cron a cada 6h)

## Desenvolvimento local

### Pré-requisitos

- Python 3.12+
- `uv` — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `pnpm` — `npm i -g pnpm`
- PostgreSQL local OU connection string Neon
- Instância de `eupeutro/hltv-api` rodando (ver https://github.com/eupeutro/hltv-api)

### Setup

```bash
# Backend
cd backend
cp .env.example .env  # ajustar DATABASE_URL
uv venv && uv pip install -e ".[dev]"
uv run alembic upgrade head
uv run uvicorn cs2_predictor.api.main:app --reload

# Pipeline (uma execução manual)
uv run python -m cs2_predictor.pipeline.run

# Frontend
cd ../frontend
cp .env.local.example .env.local
pnpm install
pnpm dev
```

### Testes

```bash
cd backend
export DATABASE_URL_TEST=postgresql+psycopg://user:pass@localhost:5432/cs2_predictor_test
uv run pytest -v
```

## Deploy

- **Pipeline**: GitHub Actions (`.github/workflows/pipeline.yml`). Secrets necessários: `DATABASE_URL`, `HLTV_API_BASE_URL`.
- **API**: Render blueprint (`backend/render.yaml`).
- **Frontend**: Vercel (`frontend/vercel.json`).

## Documentação

- Design spec: [`docs/superpowers/specs/2026-05-13-cs2-predictor-design.md`](docs/superpowers/specs/2026-05-13-cs2-predictor-design.md)
- Plano de implementação: [`docs/superpowers/plans/2026-05-13-cs2-predictor-implementation.md`](docs/superpowers/plans/2026-05-13-cs2-predictor-implementation.md)
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README with setup and deploy instructions"
```

---

## Final verification

Antes de marcar o plano como completo, rodar:

- [ ] **Step 1: Suite completa do backend**

```bash
cd backend
export DATABASE_URL_TEST=postgresql+psycopg://user:pass@localhost:5432/cs2_predictor_test
uv run pytest -v
```

Expected: todos os testes PASSED.

- [ ] **Step 2: Build do frontend**

```bash
cd frontend && pnpm build
```

Expected: build sem erros.

- [ ] **Step 3: Smoke test integrado** (manual)

Em terminais separados:
- `cd backend && uv run python -m cs2_predictor.pipeline.run` — pipeline popula o banco
- `cd backend && uv run uvicorn cs2_predictor.api.main:app --reload`
- `cd frontend && pnpm dev`
- Visitar `http://localhost:3000` e navegar pelas 5 páginas.

- [ ] **Step 4: Commit final**

```bash
git tag v0.1.0
```

---

## Notas para o executor (OpenCode + DeepSeek)

1. **Sempre invoque `superpowers:test-driven-development`** antes de implementar qualquer task com testes (Tasks 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20).
2. **Invoque `superpowers:systematic-debugging`** se um teste falhar de forma inesperada. Não pule etapas de diagnóstico.
3. **Invoque `superpowers:verification-before-completion`** antes de marcar qualquer task como done — confirme que os testes passaram, não apenas que o código foi escrito.
4. Se o teste passar antes de você escrever a implementação, **algo está errado** — investigue antes de prosseguir.
5. Commits são por task, não por step. Faça apenas no Step `Commit` de cada task.
6. **Não invente features ou refatorações** que não estejam no plano. Se algo parece ambíguo, peça esclarecimento ao invés de adivinhar.
