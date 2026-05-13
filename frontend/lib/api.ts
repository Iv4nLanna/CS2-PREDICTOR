import type {
  MatchFeatureSet,
  MatchPrediction,
  ModelAccuracyEntry,
  TeamDetail,
  TeamStats,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    next: { revalidate: 60 },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`API ${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  upcomingMatches: () => fetchJson<MatchPrediction[]>("/matches/upcoming"),
  matchPrediction: (id: number) => fetchJson<MatchPrediction>(`/matches/${id}/prediction`),
  matchFeatures: (id: number) => fetchJson<MatchFeatureSet[]>(`/matches/${id}/features`),
  listTeams: () => fetchJson<TeamStats[]>("/teams"),
  teamDetail: (id: number) => fetchJson<TeamDetail>(`/teams/${id}/stats`),
  modelAccuracy: () => fetchJson<ModelAccuracyEntry[]>("/model/accuracy"),
};
