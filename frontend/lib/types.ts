export type TeamSummary = {
  id: number;
  hltv_id: number;
  name: string;
  country: string | null;
  hltv_ranking: number | null;
};

export type MatchPrediction = {
  match_id: number;
  hltv_match_id: number;
  team_a: TeamSummary;
  team_b: TeamSummary;
  team_a_win_prob: number;
  team_b_win_prob: number;
  format: "BO1" | "BO3" | "BO5";
  is_lan: boolean;
  scheduled_at: string;
  tournament: string | null;
  model_version: string;
};

export type MatchFeatureSet = {
  team_id: number;
  win_rate_recent_decayed: number;
  head_to_head_decayed: number;
  hltv_ranking_snapshot: number | null;
  sos_score: number;
  map_stats: Record<string, number>;
};

export type TeamStats = {
  team: TeamSummary;
  recent_form: number;
  last_matches_played: number;
};

export type TeamDetail = {
  team: TeamSummary;
  recent_form: number;
  map_winrates: Record<string, number>;
  last_matches_played: number;
};

export type ModelAccuracyEntry = {
  version: string;
  trained_at: string;
  accuracy: number;
  features_used: string[];
};
