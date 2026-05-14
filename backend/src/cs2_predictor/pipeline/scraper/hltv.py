import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

SEED_TEAMS = [
    "Natus Vincere", "FaZe", "Vitality", "Spirit", "G2",
]

MATCH_TYPE_MAP = {
    "bo1": "BO1", "bo3": "BO3", "bo5": "BO5",
    "best of 1": "BO1", "best of 3": "BO3", "best of 5": "BO5",
}
MATCH_TYPE_MAP_LAN = {
    "best of 1 (lan)": "BO1", "best of 3 (lan)": "BO3", "best of 5 (lan)": "BO5",
}
for k, v in list(MATCH_TYPE_MAP_LAN.items()):
    MATCH_TYPE_MAP[k] = v


class ScraperError(Exception):
    pass


def _extract_format(match_type_str: str) -> tuple[str, bool]:
    """Extract BO format and LAN flag from match type string."""
    mt = (match_type_str or "").lower().strip()
    is_lan = "lan" in mt
    for prefix, fmt in [("best of 5", "BO5"), ("best of 3", "BO3"), ("best of 1", "BO1")]:
        if mt.startswith(prefix):
            return fmt, is_lan
    if "bo5" in mt: return "BO5", is_lan
    if "bo3" in mt: return "BO3", is_lan
    if "bo1" in mt: return "BO1", is_lan
    return "BO3", is_lan


def _ts_to_dt(timestamp_ms: float) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)


class HLTVScraper:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout, base_url=self.base_url)

    def _get(self, path: str) -> dict | list:
        try:
            response = self._client.get(path)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ScraperError(f"HLTV API request failed: {path}: {e}") from e

    def search_teams(self, query: str) -> list[dict]:
        try:
            data = self._get(f"/teams/{query}/search")
            return data.get("results", []) if isinstance(data, dict) else []
        except ScraperError:
            return []

    def get_team_profile(self, team_id: int) -> dict | None:
        try:
            data = self._get(f"/teams/{team_id}/profile")
            if isinstance(data, dict):
                return data.get("teamProfile")
        except ScraperError:
            pass
        return None

    def get_team_upcoming(self, team_id: int) -> list[dict]:
        try:
            data = self._get(f"/teams/{team_id}/upcoming-matches")
            return data.get("upcomingMatches", []) if isinstance(data, dict) else []
        except ScraperError:
            return []

    def get_team_results(self, team_id: int) -> list[dict]:
        try:
            data = self._get(f"/teams/{team_id}/results/")
            return data.get("results", []) if isinstance(data, dict) else []
        except ScraperError:
            return []

    def fetch_all_teams(self) -> list[dict]:
        """Search all seed teams and return deduplicated list.
        
        Each item: {"id": int, "name": str, "country": str, "rank": int | None}
        """
        seen: set[int] = set()
        teams: list[dict] = []
        for name in SEED_TEAMS:
            try:
                results = self.search_teams(name)
                for r in results:
                    tid = int(r["id"])
                    if tid in seen:
                        continue
                    seen.add(tid)
                    rank = None
                    try:
                        profile = self.get_team_profile(tid)
                        if profile:
                            rank = profile.get("worldRanking")
                    except ScraperError:
                        pass
                    teams.append({
                        "id": tid,
                        "name": r.get("name", name),
                        "country": r.get("country"),
                        "rank": rank,
                    })
            except ScraperError as e:
                logger.warning("Failed to search team %s: %s", name, e)
        return teams

    def normalize_upcoming_matches(
        self, team_id: int, raw_matches: list[dict]
    ) -> list[dict]:
        """Convert raw API upcoming matches to internal format.
        
        Internal format: {"id", "team_a_id", "team_b_id", "format", "is_lan",
                          "map_pool": [], "tournament", "scheduled_at"}
        """
        result = []
        for m in raw_matches:
            fmt, is_lan = _extract_format(m.get("matchType", ""))
            ts = m.get("matchTimestamp", 0)
            result.append({
                "id": int(m["matchId"]),
                "team_a_id": team_id,
                "team_b_id": int(m["rivalTeamId"]),
                "format": fmt,
                "is_lan": is_lan,
                "map_pool": [],
                "tournament": m.get("eventName"),
                "scheduled_at": _ts_to_dt(ts).isoformat() if ts else None,
            })
        return result

    def normalize_results(
        self, team_id: int, raw_results: list[dict], team_name_to_id: dict[str, int]
    ) -> list[dict]:
        """Convert raw API results to internal format.
        
        Internal format: {"id", "winner_id", "score_detail": {}, "played_at"}
        
        Since results have team names but not always team IDs, we use team_name_to_id.
        team_id is the current team; the API gives results from this team's perspective.
        """
        result = []
        for r in raw_results:
            t1_name = r.get("team1Name", "")
            t2_name = r.get("team2Name", "")
            t1_id = team_name_to_id.get(t1_name)
            t2_id = team_name_to_id.get(t2_name)
            if t1_id is None or t2_id is None:
                if t1_name in team_name_to_id:
                    t1_id = team_name_to_id[t1_name]
                    continue
                elif t2_name in team_name_to_id:
                    t2_id = team_name_to_id[t2_name]
                    continue
                else:
                    continue

            score_detail = {}
            if r.get("team1Score") is not None and r.get("team2Score") is not None:
                score_detail["score"] = [r["team1Score"], r["team2Score"]]

            match_won = r.get("matchWon", False)
            if team_id == t1_id:
                winner_id = t1_id if match_won else t2_id
            else:
                winner_id = t2_id if match_won else t1_id

            played_at_str = r.get("matchDate", "")
            if played_at_str:
                played_dt = datetime.fromisoformat(played_at_str).replace(tzinfo=timezone.utc)
            else:
                played_dt = datetime.now(timezone.utc)

            result.append({
                "id": int(r["matchId"]),
                "winner_id": winner_id,
                "score_detail": score_detail,
                "played_at": played_dt.isoformat(),
            })
        return result

    def close(self):
        self._client.close()
