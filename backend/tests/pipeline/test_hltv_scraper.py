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
