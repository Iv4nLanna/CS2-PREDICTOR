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
