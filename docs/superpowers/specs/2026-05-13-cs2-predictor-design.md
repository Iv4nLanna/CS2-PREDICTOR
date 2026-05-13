# CS2 Win Predictor — Design Spec

**Date:** 2026-05-13
**Status:** Approved for planning
**Owner:** Ivan Lana

---

## 1. Overview

Plataforma web que prevê estatisticamente qual time profissional de Counter-Strike 2 tem maior probabilidade de vencer em partidas agendadas. Combina dados públicos do HLTV com um modelo de regressão logística calibrada para gerar probabilidades pré-partida transparentes e auditáveis.

## 2. Goals

- Prever probabilidade de vitória pré-partida para confrontos profissionais de CS2.
- Expor previsões via web app público com dashboard, detalhe de partida e perfil de time.
- Manter transparência: o usuário vê quais features alimentaram cada previsão e a acurácia histórica do modelo.
- Atender tanto apostadores (comparação com odds) quanto fãs/analistas (entendimento competitivo).
- Operar em infraestrutura de custo zero ou baixo na v1.

## 3. Non-Goals (v1)

- Previsão ao vivo durante partidas (por round/mapa). Arquitetura deve permitir adicionar depois, mas não é entregue na v1.
- Integração com Faceit API. Modelagem de dados deve acomodar Faceit como expansão futura.
- Sistema de contas/autenticação.
- Sistema de apostas integrado ou venda de picks.
- Testes de UI/frontend.
- App mobile nativo.

## 4. Target Audience

Plataforma pública direcionada a dois grupos:

- **Apostadores** — comparam as probabilidades calculadas com odds de casas de aposta para identificar valor.
- **Fãs e analistas** — usam o produto para entender o cenário competitivo, forma de times e força relativa.

A página de transparência do modelo (acurácia histórica, calibração) é especialmente importante para o primeiro grupo construir confiança.

## 5. Architecture

Sistema de 4 camadas. O pipeline de dados é **desacoplado** do servidor web: o pipeline calcula previsões e grava no banco, a API apenas lê.

```
┌─────────────────────────────────────────────────────┐
│                   PIPELINE (agendado)                │
│                                                      │
│  HLTV API → Scraper → Feature Engineering → Model   │
│                               ↓                     │
│                         PostgreSQL                   │
│                    (matches, teams,                  │
│                     features, predictions)           │
└──────────────────────────┬──────────────────────────┘
                           │ lê
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │  (read-only)│
                    └──────┬──────┘
                           │ REST/JSON
                    ┌──────▼──────┐
                    │   Next.js   │
                    │  (SSR/CSR)  │
                    └─────────────┘
```

### 5.1 Camadas

| Camada | Responsabilidade | Tecnologia |
|---|---|---|
| **Scraper** | Coletar dados do HLTV (times, rankings, partidas, resultados) | Python, cliente HTTP da `eupeutro/hltv-api` |
| **Feature Engineering** | Transformar dados brutos em features para o modelo | Python, pandas |
| **Modelo** | Treinar regressão logística, calibrar com Platt Scaling, gerar previsões | scikit-learn |
| **API** | Servir previsões e dados ao frontend, sem escrita | FastAPI |
| **Frontend** | Renderizar dashboard, detalhe de partida, perfil de time, transparência | Next.js (App Router) |

### 5.2 Decisões arquiteturais

- **Previsões pré-computadas**: o pipeline calcula e grava todas as previsões; a API nunca executa o modelo on-demand. Isso simplifica a API, elimina latência de inferência e separa preocupações.
- **Pipeline em processo separado**: roda como GitHub Action agendado, não dentro do servidor FastAPI. Trava no pipeline não derruba a API.
- **API read-only**: a única origem de escrita no banco é o pipeline. Reduz risco de inconsistência e permite escalar API horizontalmente no futuro.

## 6. Data Sources

### 6.1 Fonte primária — HLTV via `eupeutro/hltv-api`

API REST Python (FastAPI) que faz scraping do HLTV.org com bypass de Cloudflare via FlareSolverr.

- Endpoints usados: times, jogadores, ranking, partidas agendadas, partidas finalizadas, eventos.
- **Risco conhecido**: dependência de scraping — pode quebrar se o HLTV mudar layout. Mitigação: o pipeline tolera falhas mantendo previsões anteriores.

### 6.2 Expansão futura — Faceit API

Não consumida na v1. O modelo de dados (ver §8) reserva colunas opcionais em `players` (`faceit_rating`, `faceit_id`) para que adicionar Faceit no futuro não exija migração disruptiva.

## 7. Feature Engineering

O modelo é uma regressão logística, portanto **linear**: qualidade da previsão depende quase inteiramente da qualidade das features. A camada de Feature Engineering computa features por **partida** (não por time isolado), capturando o estado dos dois times no momento do confronto.

### 7.1 Features de v1

| Feature | Descrição | Tipo |
|---|---|---|
| `win_rate_recent_decayed` | Win rate com decaimento exponencial: vitórias recentes pesam mais. Captura forma atual, mudanças de roster, slumps. | float |
| `win_rate_per_map` | Win rate de cada time em cada mapa do pool veto. CS2 tem veto, win rate geral esconde força específica. | jsonb |
| `head_to_head_decayed` | Histórico de confrontos diretos com decaimento por tempo. Confrontos antigos com roster diferente pesam menos. | float |
| `hltv_ranking_diff` | Diferença de ranking HLTV entre os times. Agrega muito histórico em um número. | int |
| `sos_score` | Strength of Schedule. Win rate ponderado pela força dos adversários (vitória contra top-10 pesa mais que contra top-50). Evita inflar times de regiões fracas. | float |
| `format` | BO1, BO3 ou BO5. BO1 tem muito mais variância — underdog ganha mais. | categórica |
| `is_lan` | LAN vs Online. Alguns times performam consistentemente melhor presencialmente. | bool |

### 7.2 Pipeline de features

```
dados brutos → normalização → features por mapa
                            → forma com decaimento
                            → SOS (força de adversários)
                            → features de contexto (BO, LAN)
                            → regressão logística
                            → calibração (Platt Scaling)
                            → probabilidade salva no banco
```

### 7.3 Calibração

Regressão logística pura pode produzir probabilidades mal calibradas (ex: prever "70%" quando a frequência real é 60%). Aplicar **Platt Scaling** sobre as previsões corrige isso. É o que separa "probabilidade" de "score" — crítico para apostadores compararem com odds.

## 8. Data Model

PostgreSQL como banco primário. Tabelas:

### `teams`
- `id` (PK)
- `hltv_id` (unique)
- `name`
- `country`
- `hltv_ranking`
- `updated_at`

### `players`
- `id` (PK)
- `team_id` (FK → teams)
- `hltv_id`
- `name`
- `rating`
- `role`
- `active`
- `faceit_rating` (nullable, reservado para expansão)
- `faceit_id` (nullable, reservado para expansão)

### `matches`
- `id` (PK)
- `hltv_id` (unique)
- `team_a_id` (FK → teams)
- `team_b_id` (FK → teams)
- `format` (enum: BO1, BO3, BO5)
- `is_lan` (bool)
- `map_pool` (jsonb)
- `tournament`
- `scheduled_at`
- `status` (enum: scheduled, live, finished, cancelled)
- `created_at`

### `match_results`
- `id` (PK)
- `match_id` (FK → matches, unique)
- `winner_id` (FK → teams)
- `score_detail` (jsonb — placar por mapa)
- `played_at`

### `team_features`
- `id` (PK)
- `team_id` (FK → teams)
- `match_id` (FK → matches)
- `win_rate_recent_decayed` (float)
- `head_to_head_decayed` (float) — em relação ao adversário na mesma partida
- `hltv_ranking_snapshot` (int) — ranking no momento do confronto
- `sos_score` (float)
- `map_stats` (jsonb) — win rate por mapa do `map_pool` da partida
- `computed_at`

Computada **por partida**, não por time absoluto, para capturar o estado no momento do confronto.

Features contextuais (`format`, `is_lan`) ficam na tabela `matches` e são lidas diretamente pelo modelo, não duplicadas em `team_features`.

### `predictions`
- `id` (PK)
- `match_id` (FK → matches)
- `team_a_win_prob` (float, 0–1)
- `team_b_win_prob` (float, 0–1)
- `model_version` (FK → model_runs.version)
- `calibrated` (bool)
- `created_at`

### `model_runs`
- `id` (PK)
- `version` (unique)
- `trained_at`
- `accuracy` (acurácia em validação temporal)
- `features_used` (jsonb)

### Decisões

- `map_stats` e `features_used` como **JSONB** — permite adicionar features sem migração de schema.
- `model_runs` rastreia cada retreinamento com acurácia — permite comparar versões e exibir histórico na página de transparência.
- Faceit cabe como colunas opcionais em `players` — sem quebrar nada.

## 9. API (FastAPI)

Todos os endpoints são **read-only**. A API nunca escreve no banco — escrita é exclusividade do pipeline.

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/matches/upcoming` | Próximas partidas com probabilidades calculadas |
| `GET` | `/matches/{id}/prediction` | Previsão detalhada de uma partida específica |
| `GET` | `/matches/{id}/features` | Features usadas na previsão (transparência) |
| `GET` | `/teams` | Listagem de times com stats resumidos |
| `GET` | `/teams/{id}/stats` | Stats e histórico detalhado de um time |
| `GET` | `/model/accuracy` | Acurácia histórica do modelo por versão |

## 10. Frontend (Next.js)

App Router. SSR nas páginas principais para SEO e first-paint rápido; carregamento client-side em dados de detalhe.

| Rota | Renderização | Descrição |
|---|---|---|
| `/` | SSR | Dashboard: próximas partidas com % de vitória |
| `/matches/[id]` | SSR | Detalhe da partida: probabilidades, features, mapa a mapa |
| `/teams` | CSR | Ranking de times com stats resumidos |
| `/teams/[id]` | CSR | Perfil do time: forma recente, win rate por mapa |
| `/model` | SSR | Transparência: acurácia histórica, versão atual, calibração |

A página `/model` é o diferencial para apostadores — exibe acurácia, distribuição de calibração, total de partidas previstas. Constrói confiança no produto.

## 11. Pipeline

Processo Python separado do servidor web. Executado como **GitHub Action agendado** (cron a cada 6 horas; intervalo configurável via env var).

### 11.1 Fluxo

1. Scraper coleta dados da HLTV API (times, rankings, partidas agendadas, partidas finalizadas desde a última execução).
2. Persiste dados brutos atualizados em `teams`, `players`, `matches`, `match_results`.
3. Feature Engineering computa features por partida (registros em `team_features`).
4. Se há **≥10 novas partidas finalizadas** desde o último treino, retreina o modelo; senão, mantém versão atual.
5. Aplica Platt Scaling sobre as previsões.
6. Grava previsões em `predictions` com `model_version`.
7. Registra `model_runs` com acurácia avaliada em validação temporal.

### 11.2 Error Handling

- **HLTV API indisponível** → pipeline registra erro, mantém previsões anteriores no banco, não derruba a API.
- **Dados insuficientes para retreinar** (<10 novas partidas) → mantém modelo atual, não sobrescreve.
- **Feature inválida ou faltante** para uma partida → essa partida é marcada como `sem previsão`. O frontend exibe estado explícito; nunca inventa probabilidade.

## 12. Hosting

Setup target: custo zero na v1.

| Componente | Serviço | Justificativa |
|---|---|---|
| Pipeline | GitHub Actions (cron) | 2.000 min/mês free. 4 runs/dia × ~5min = ~600 min/mês. |
| PostgreSQL | Neon (preferência) ou Supabase | 500MB–1GB free, suficiente para o volume previsto. |
| FastAPI | Render free | Dorme após 15min, cold start ~30s. Aceitável: dados já estão no banco. |
| Frontend | Vercel | Free tier sem limitação prática para o caso. |

Migração futura para Railway Hobby ($5/mês) elimina cold start do FastAPI se necessário.

## 13. Testing

| Camada | Estratégia |
|---|---|
| **Feature Engineering** | Testes unitários por feature (forma decayed, SOS, win rate por mapa) com dados mockados. |
| **Modelo** | **Validação temporal**: treina em dados até mês X, valida em mês X+1. Previne data leakage. |
| **API** | Testes de integração com banco de teste real (não mockado). |
| **Frontend** | Sem testes de UI na v1. |

Mock de banco é proibido em testes de integração da API — testes precisam refletir comportamento real do Postgres.

## 14. Future Expansions

- **Faceit API**: adicionar como segunda fonte. Tabelas já reservam colunas (`faceit_rating`, `faceit_id`).
- **Previsão ao vivo**: pipeline separado consumindo dados por round/mapa. Modelo de dados já permite (basta novo tipo de `prediction`).
- **Histórico do usuário**: salvar partidas favoritas, alertas. Requer autenticação — fora do escopo da v1.
- **Open data / API pública**: expor endpoints de leitura como produto independente para integradores.

## 15. Glossary

- **HLTV**: hltv.org, site de referência do cenário profissional de CS.
- **BO1/BO3/BO5**: Best of 1/3/5. Formato da partida.
- **Veto**: processo de seleção e exclusão de mapas antes da partida.
- **SOS**: Strength of Schedule. Métrica que pondera vitórias pela força dos adversários.
- **Platt Scaling**: técnica de pós-processamento que calibra probabilidades de classificadores binários.
- **Validação temporal**: separação treino/teste por tempo (não aleatória) para prever performance em produção.
