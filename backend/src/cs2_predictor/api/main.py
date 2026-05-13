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
