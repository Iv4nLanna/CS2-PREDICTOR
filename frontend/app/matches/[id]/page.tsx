import { notFound } from "next/navigation";

import { api } from "@/lib/api";
import { ProbabilityBar } from "@/components/ProbabilityBar";

export const revalidate = 60;

type FeatureRowProps = { label: string; value: number | string | null };
function FeatureRow({ label, value }: FeatureRowProps) {
  return (
    <div className="flex justify-between border-b border-zinc-800 py-1 text-sm">
      <span className="text-zinc-400">{label}</span>
      <span>{value ?? "—"}</span>
    </div>
  );
}

export default async function MatchDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const matchId = Number(id);
  if (Number.isNaN(matchId)) notFound();

  let prediction;
  let features;
  try {
    [prediction, features] = await Promise.all([
      api.matchPrediction(matchId),
      api.matchFeatures(matchId),
    ]);
  } catch {
    notFound();
  }

  const teamA = features.find((f) => f.team_id === prediction.team_a.id);
  const teamB = features.find((f) => f.team_id === prediction.team_b.id);

  return (
    <div className="mx-auto max-w-3xl">
      <p className="mb-1 text-sm text-zinc-400">
        {prediction.tournament ?? "—"} · {prediction.format} ·{" "}
        {prediction.is_lan ? "LAN" : "Online"}
      </p>
      <h1 className="mb-6 text-2xl font-semibold">
        {prediction.team_a.name} <span className="text-zinc-500">vs</span> {prediction.team_b.name}
      </h1>
      <div className="mb-8">
        <ProbabilityBar
          leftLabel={prediction.team_a.name}
          rightLabel={prediction.team_b.name}
          leftProb={prediction.team_a_win_prob}
        />
        <p className="mt-2 text-xs text-zinc-500">
          Modelo {prediction.model_version} · agendado para{" "}
          {new Date(prediction.scheduled_at).toLocaleString("pt-BR")}
        </p>
      </div>

      <h2 className="mb-3 text-lg font-semibold">Features</h2>
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        {[
          { team: prediction.team_a, feats: teamA },
          { team: prediction.team_b, feats: teamB },
        ].map(({ team, feats }) => (
          <div key={team.id} className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <h3 className="mb-2 font-semibold">{team.name}</h3>
            <FeatureRow label="Forma recente (decay)" value={feats?.win_rate_recent_decayed.toFixed(2) ?? null} />
            <FeatureRow label="H2H (decay)" value={feats?.head_to_head_decayed.toFixed(2) ?? null} />
            <FeatureRow label="Ranking HLTV (snapshot)" value={feats?.hltv_ranking_snapshot ?? null} />
            <FeatureRow label="SOS" value={feats?.sos_score.toFixed(2) ?? null} />
            {feats && Object.entries(feats.map_stats).map(([map, rate]) => (
              <FeatureRow key={map} label={`Win rate ${map}`} value={rate.toFixed(2)} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
