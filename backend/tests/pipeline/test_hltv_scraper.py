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


def test_search_teams():
    scraper = HLTVScraper(base_url="http://fake")
    payload = {"results": [{"id": "4608", "name": "Navi", "country": "EU"}]}
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        result = scraper.search_teams("Navi")
    assert len(result) == 1
    assert result[0]["name"] == "Navi"


def test_search_teams_empty_on_missing():
    scraper = HLTVScraper(base_url="http://fake")
    with patch("httpx.Client.get", return_value=_mock_response({"results": []})):
        result = scraper.search_teams("Unknown")
    assert result == []


def test_get_team_profile():
    scraper = HLTVScraper(base_url="http://fake")
    payload = {"teamProfile": {"name": "Navi", "worldRanking": 1}}
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        profile = scraper.get_team_profile(4608)
    assert profile["name"] == "Navi"
    assert profile["worldRanking"] == 1


def test_get_team_upcoming():
    scraper = HLTVScraper(base_url="http://fake")
    payload = {
        "upcomingMatches": [{
            "matchId": "2394171", "rivalTeamName": "Legacy", "rivalTeamId": "12468",
            "matchType": "Best of 3 (LAN)", "matchTimestamp": 1778715000000.0,
            "eventName": "IEM Atlanta",
        }]
    }
    with patch("httpx.Client.get", return_value=_mock_response(payload)):
        matches = scraper.get_team_upcoming(4608)
    assert len(matches) == 1
    assert matches[0]["matchId"] == "2394171"


def test_normalize_upcoming():
    scraper = HLTVScraper(base_url="http://fake")
    raw = [{
        "matchId": "2394171", "rivalTeamName": "Legacy", "rivalTeamId": "12468",
        "matchType": "Best of 3 (LAN)", "matchTimestamp": 1778715000000.0,
        "eventName": "IEM Atlanta",
    }]
    result = scraper.normalize_upcoming_matches(4608, raw)
    assert len(result) == 1
    assert result[0]["id"] == 2394171
    assert result[0]["team_a_id"] == 4608
    assert result[0]["team_b_id"] == 12468
    assert result[0]["format"] == "BO3"
    assert result[0]["is_lan"] is True


def test_normalize_results():
    scraper = HLTVScraper(base_url="http://fake")
    raw = [{
        "matchId": "2394161",
        "team1Name": "Navi", "team1Score": 2,
        "team2Name": "Legacy", "team2Score": 1,
        "matchType": "bo3", "matchDate": "2026-05-12",
        "matchWon": True,
    }]
    name_to_id = {"Navi": 4608, "Legacy": 12468}
    result = scraper.normalize_results(4608, raw, name_to_id)
    assert len(result) == 1
    assert result[0]["id"] == 2394161
    assert result[0]["winner_id"] == 4608  # Navi won
    assert "played_at" in result[0]


def test_raises_on_http_error():
    scraper = HLTVScraper(base_url="http://fake")
    with patch("httpx.Client.get", return_value=_mock_response({}, status_code=500)):
        with pytest.raises(ScraperError):
            scraper.search_teams("Navi")
