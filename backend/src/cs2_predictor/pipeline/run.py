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
