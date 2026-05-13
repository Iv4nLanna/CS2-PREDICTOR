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
from cs2_predictor.pipeline.scraper.hltv import HLTVScraper, ScraperError
from cs2_predictor.pipeline.scraper.persistence import (
    upsert_match_results,
    upsert_matches,
    upsert_teams,
)

logger = logging.getLogger(__name__)


def _normalize_match_format(match_type_str: str) -> str:
    """Convert API match type string to BO format."""
    mt = (match_type_str or "").lower().strip()
    is_lan = "lan" in mt
    for prefix, fmt in [("best of 5", "BO5"), ("best of 3", "BO3"), ("best of 1", "BO1")]:
        if mt.startswith(prefix):
            return fmt
    if "bo5" in mt: return "BO5"
    if "bo3" in mt: return "BO3"
    if "bo1" in mt: return "BO1"
    return "BO3"


def _next_version() -> str:
    return f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def run_pipeline(session: Session | None = None, min_matches_to_retrain: int | None = None) -> dict:
    from cs2_predictor.pipeline.scraper.hltv import (
        SEED_TEAMS,
        MATCH_TYPE_MAP,
    )

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
            # Phase 1: Discover and upsert teams
            teams = scraper.fetch_all_teams()
            if not teams:
                raise ScraperError("no teams discovered from HLTV API")
            upsert_teams(session, teams)
            session.commit()

            # Build team name -> ID map from the session
            from cs2_predictor.db.models import Team
            all_teams = session.query(Team).all()
            team_name_to_id: dict[str, int] = {}
            for t in all_teams:
                if t.name:
                    team_name_to_id[t.name] = t.hltv_id

            # Phase 2: Fetch upcoming matches and results per team
            seen_upcoming: set[int] = set()
            seen_results: set[int] = set()
            all_upcoming: list[dict] = []
            all_results: list[dict] = []

            for team in teams:
                tid = team["id"]
                try:
                    # Upcoming
                    raw_upcoming = scraper.get_team_upcoming(tid)
                    for m in scraper.normalize_upcoming_matches(tid, raw_upcoming):
                        if m["id"] not in seen_upcoming:
                            seen_upcoming.add(m["id"])
                            all_upcoming.append(m)
                except Exception as e:
                    logger.warning("Failed to get upcoming for team %d: %s", tid, e)

                try:
                    # Results
                    raw_results = scraper.get_team_results(tid)
                    for m in scraper.normalize_results(tid, raw_results, team_name_to_id):
                        if m["id"] not in seen_results:
                            seen_results.add(m["id"])
                            all_results.append(m)
                except Exception as e:
                    logger.warning("Failed to get results for team %d: %s", tid, e)

            if all_upcoming:
                upsert_matches(session, all_upcoming)
            if all_results:
                upsert_match_results(session, all_results)
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
                    from cs2_predictor.db.models import ModelRun
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
