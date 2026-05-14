# CS2 Win Predictor

> **Status: Em desenvolvimento — não funcional no momento.**
> O HLTV.org bloqueia scrapers automatizados (Cloudflare). Pendente de integração com FACEIT API ou outra fonte de dados confiável.

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
