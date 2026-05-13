import Link from "next/link";

import type { MatchPrediction } from "@/lib/types";
import { ProbabilityBar } from "./ProbabilityBar";

function formatDate(iso: string) {
  return new Date(iso).toLocaleString("pt-BR", {
    day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

export function MatchCard({ match }: { match: MatchPrediction }) {
  return (
    <Link
      href={`/matches/${match.match_id}`}
      className="block rounded-lg border border-zinc-800 bg-zinc-900 p-4 hover:border-zinc-700"
    >
      <div className="mb-2 flex items-center justify-between text-xs text-zinc-400">
        <span>{match.tournament ?? "—"}</span>
        <span>{match.format} {match.is_lan ? "· LAN" : "· Online"}</span>
      </div>
      <div className="mb-3 flex items-center justify-between font-semibold">
        <span>{match.team_a.name}</span>
        <span className="text-xs text-zinc-500">{formatDate(match.scheduled_at)}</span>
        <span>{match.team_b.name}</span>
      </div>
      <ProbabilityBar
        leftLabel={match.team_a.name}
        rightLabel={match.team_b.name}
        leftProb={match.team_a_win_prob}
      />
    </Link>
  );
}
