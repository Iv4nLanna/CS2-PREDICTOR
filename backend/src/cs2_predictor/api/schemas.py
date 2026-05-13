from datetime import datetime

from pydantic import BaseModel


class TeamSummary(BaseModel):
    id: int
    hltv_id: int
    name: str
    country: str | None
    hltv_ranking: int | None


class MatchPrediction(BaseModel):
    match_id: int
    hltv_match_id: int
    team_a: TeamSummary
    team_b: TeamSummary
    team_a_win_prob: float
    team_b_win_prob: float
    format: str
    is_lan: bool
    scheduled_at: datetime
    tournament: str | None
    model_version: str


class MatchFeatureSet(BaseModel):
    team_id: int
    win_rate_recent_decayed: float
    head_to_head_decayed: float
    hltv_ranking_snapshot: int | None
    sos_score: float
    map_stats: dict[str, float]


class MatchDetail(BaseModel):
    prediction: MatchPrediction
    features: list[MatchFeatureSet]


class ModelAccuracyEntry(BaseModel):
    version: str
    trained_at: datetime
    accuracy: float
    features_used: list[str]


class TeamStats(BaseModel):
    team: TeamSummary
    recent_form: float
    last_matches_played: int


class TeamDetail(BaseModel):
    team: TeamSummary
    recent_form: float
    map_winrates: dict[str, float]
    last_matches_played: int
