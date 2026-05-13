import Link from "next/link";

import { api } from "@/lib/api";

export const revalidate = 300;

export default async function TeamsPage() {
  let teams: Awaited<ReturnType<typeof api.listTeams>> = [];
  try {
    teams = await api.listTeams();
  } catch {
    return <p className="text-rose-300">Não foi possível carregar os times.</p>;
  }

  return (
    <div className="mx-auto max-w-3xl">
      <h1 className="mb-6 text-2xl font-semibold">Times</h1>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-zinc-400">
            <th className="py-2">Ranking</th>
            <th className="py-2">Time</th>
            <th className="py-2">País</th>
            <th className="py-2">Forma recente</th>
            <th className="py-2">Partidas</th>
          </tr>
        </thead>
        <tbody>
          {teams.map((t) => (
            <tr key={t.team.id} className="border-t border-zinc-800">
              <td className="py-2">{t.team.hltv_ranking ?? "—"}</td>
              <td className="py-2">
                <Link href={`/teams/${t.team.id}`} className="hover:text-emerald-400">
                  {t.team.name}
                </Link>
              </td>
              <td className="py-2 text-zinc-400">{t.team.country ?? "—"}</td>
              <td className="py-2">{(t.recent_form * 100).toFixed(0)}%</td>
              <td className="py-2 text-zinc-400">{t.last_matches_played}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
