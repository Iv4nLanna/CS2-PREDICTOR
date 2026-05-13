from unittest.mock import MagicMock, patch

from cs2_predictor.pipeline.run import run_pipeline


def test_run_pipeline_continues_on_scraper_failure(db_session):
    scraper = MagicMock()
    scraper.fetch_all_teams.side_effect = Exception("HLTV down")
    with patch("cs2_predictor.pipeline.run.HLTVScraper", return_value=scraper):
        result = run_pipeline(session=db_session)
    assert result["status"] == "partial"
    assert "scraper_error" in result["errors"]


def test_run_pipeline_skips_retraining_below_threshold(db_session):
    scraper = MagicMock()
    scraper.fetch_all_teams.return_value = []
    with patch("cs2_predictor.pipeline.run.HLTVScraper", return_value=scraper):
        result = run_pipeline(session=db_session, min_matches_to_retrain=10)
    assert result["retrained"] is False
