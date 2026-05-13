import { notFound } from "next/navigation";

import { api } from "@/lib/api";

export const revalidate = 300;

export default async function TeamDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const teamId = Number(id);
  if (Number.isNaN(teamId)) notFound();

  let detail;
  try {
    detail = await api.teamDetail(teamId);
  } catch {
    notFound();
  }

  return (
    <div className="mx-auto max-w-3xl">
      <p className="mb-1 text-sm text-zinc-400">
        {detail.team.country ?? "—"} · Ranking HLTV {detail.team.hltv_ranking ?? "—"}
      </p>
      <h1 className="mb-6 text-2xl font-semibold">{detail.team.name}</h1>

      <div className="mb-8 grid grid-cols-2 gap-4">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-400">Forma recente</p>
          <p className="text-2xl font-semibold">{(detail.recent_form * 100).toFixed(0)}%</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-400">Partidas registradas</p>
          <p className="text-2xl font-semibold">{detail.last_matches_played}</p>
        </div>
      </div>

      <h2 className="mb-3 text-lg font-semibold">Win rate por mapa</h2>
      <div className="space-y-1">
        {Object.entries(detail.map_winrates).map(([map, rate]) => (
          <div key={map} className="flex items-center justify-between border-b border-zinc-800 py-1 text-sm">
            <span className="text-zinc-400">{map}</span>
            <span>{(rate * 100).toFixed(0)}%</span>
          </div>
        ))}
        {Object.keys(detail.map_winrates).length === 0 && (
          <p className="text-zinc-400">Sem histórico por mapa ainda.</p>
        )}
      </div>
    </div>
  );
}
