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
